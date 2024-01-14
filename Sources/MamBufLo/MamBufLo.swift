import Metal

public struct MBLStateDictMetadata: Decodable
{
    public var stateDictKey: String
    public var shape: [UInt32]
    public var baseStride: Int
    
    enum CodingKeys : String, CodingKey 
    {
        case key
        case shape
        case is16bit
    }
    
    public init(from decoder: Decoder) throws 
    {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        
        stateDictKey = try values.decode(String.self, forKey: .key)
        shape        = try values.decode(Array<UInt32>.self, forKey: .shape)
        
        baseStride = try values.decode(Bool.self, forKey: .is16bit)
        ? MemoryLayout<Float16>.stride
        : MemoryLayout<Float32>.stride
    }
}


public struct MBLEntrySpec<ModelSpec: MBLModelSpec, KeyProvider> {
    public var swiftKey: PartialKeyPath<KeyProvider>
    public var pythonKey: String
    public var shape: [KeyPath<ModelSpec.HParams, UInt32>]
    public init(key: PartialKeyPath<KeyProvider>, pythonKey: String, shape: [KeyPath<ModelSpec.HParams, UInt32>]) {
        self.swiftKey = key
        self.pythonKey = pythonKey
        self.shape = shape
    }
}


public protocol MBLModelSpec<HParams> {
    /// A type describing hyperparameters, which define some properties of the model parameters
    associatedtype HParams
    associatedtype BaseKeys
    associatedtype LayerKeys
    associatedtype BaseSpecs:  Sequence<MBLEntrySpec<Self, BaseKeys>>
    associatedtype LayerSpecs: Sequence<MBLEntrySpec<Self, LayerKeys>>
    
    /// The name of a model folder on disk
    var name: String { get }
    var hp: HParams { get }
    
    var baseSpecs:  BaseSpecs  { get }
    var layerSpecs: LayerSpecs { get }
}

extension MBLModelSpec
{
    public func findStateDictSpec(for sdName: String) throws -> MBLEntrySpec<Self, BaseKeys> {
        guard let match = baseSpecs.first(where: { sdName.contains($0.pythonKey) }) else {
            throw MamBufLoError.unknownLayer("MBLModelSpec.findStateShapeEntry: Nothing in spec matches within \(sdName)")
        }
        return match
    }
    
    public func findStateDictSpec(for sdName: String, layerNumber: UInt32) throws -> MBLEntrySpec<Self, LayerKeys> {
        guard let match = layerSpecs.first(where: { sdName.contains($0.pythonKey) }) else {
            throw MamBufLoError.unknownLayer("MBLModelSpec.findStateShapeEntry: Nothing in spec matches within \(sdName)")
        }
        return match
    }
    
    public func validateShapeToPlan(for metadata: MBLStateDictMetadata) throws -> [UInt32] {
        let match = try findStateDictSpec(for: metadata.stateDictKey)
        let plannedShape = match.shape.map { hp[keyPath: $0] }
        guard metadata.shape == plannedShape else {
            let actual   = "\(metadata.stateDictKey): \(metadata.shape.debugDescription)"
            let expected = "\(match.shape): \(plannedShape)"
            throw MamBufLoError.invalidParameterShape("\(actual) is not according to plan \(expected)")
        }

        return plannedShape
    }
    
    public func validateShapeToPlan(for metadata: MBLStateDictMetadata, layerNumber: UInt32) throws -> [UInt32] {
        let match = try findStateDictSpec(for: metadata.stateDictKey, layerNumber: layerNumber)
        let plannedShape = match.shape.map { hp[keyPath: $0] }
        guard metadata.shape == plannedShape else {
            let actual   = "\(metadata.stateDictKey): \(metadata.shape.debugDescription)"
            let expected = "\(match.shape): \(plannedShape)"
            throw MamBufLoError.invalidParameterShape("\(actual) is not according to plan \(expected)")
        }

        return plannedShape
    }
}

public struct MBLStateDictEntryDescriptor: Hashable, Equatable
{
    public let metadata: MBLStateDictMetadata
    public let dataPath: String
    public var layerNumber: UInt32?
    
    public var elemCount: Int { Int(metadata.shape.reduce(1, *)) }
    public var bytesExpected: Int { elemCount * metadata.baseStride  }
    public var prettyName: String {
        metadata.stateDictKey
        + "/"
        + (layerNumber?.formatted() ?? "base")
    }
    
    public init(_ stateEntryPath: String) throws {
        let metadataPath = stateEntryPath + "/metadata.json"
        let metadataJson = try String(contentsOfFile: metadataPath)
        let metadata = try JSONDecoder().decode(MBLStateDictMetadata.self, from: metadataJson.data(using: .utf8)!)
        self.metadata = metadata
        self.dataPath = stateEntryPath + "/weights.bin"
        self.layerNumber = try? UInt32(String(/layers\.([0-9]+)/.firstMatch(in: metadata.stateDictKey)?.output.1 ?? ""))
        try validateBytesExpected()
    }
    
    public func validateBytesExpected() throws {
        let atts = try FileManager.default.attributesOfItem(atPath: dataPath)
        let fileSize = atts[.size] as? Int
        guard fileSize == bytesExpected else {
            let desc = "\(metadata.stateDictKey): shape \(metadata.shape)"
            let sizeEx = bytesExpected.formatted(.byteCount(style: .file))
            let sizeAc = fileSize?.formatted(.byteCount(style: .file)) ?? "unknown"
            throw MamBufLoError.filesizeMismatch("\(desc) implies a filesize of \(sizeEx), but actual filesize is \(sizeAc)")
        }
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(metadata.stateDictKey)
    }
    
    public static func == (lhs: MBLStateDictEntryDescriptor, rhs: MBLStateDictEntryDescriptor) -> Bool {
        lhs.hashValue == rhs.hashValue
    }
}


enum MamBufLoError: Error {
    case invalidFile(String),
         incompleteState(String),
         filesizeMismatch(String),
         invalidParameter(String),
         invalidParameterShape(String),
         missingBase(String),
         extraneousStateEntry(String),
         missingLayer(UInt32),
         incompleteLayer(String),
         unknownLayer(String),
         failedToMakeCommandBuffer,
         failedToMakeCommandEncoder,
         failedToMakeMetalBuffer,
         failedToMakeMetalHeap,
         internalError
}

public struct MBLHeapedParameter {
    public var data: MTLBuffer
    public var desc: MBLStateDictEntryDescriptor
    public lazy var name: String = { desc.prettyName }()
    
    public lazy var argu: MTLArgumentDescriptor = {
        let argu = MTLArgumentDescriptor()
        
        argu.access = .readOnly
        argu.dataType = .half
        argu.arrayLength = desc.elemCount
        return argu
    }()
    
    
    public func getKeyPath<ModelSpec: MBLModelSpec>(within modelSpec: ModelSpec) throws
    -> PartialKeyPath<ModelSpec.BaseKeys> {
        return try modelSpec.findStateDictSpec(for: desc.metadata.stateDictKey).swiftKey
    }
    
    public func getKeyPath<ModelSpec: MBLModelSpec>(within modelSpec: ModelSpec) throws
    -> PartialKeyPath<ModelSpec.LayerKeys> {
        return try modelSpec.findStateDictSpec(for: desc.metadata.stateDictKey, layerNumber: desc.layerNumber!).swiftKey
    }
}


@dynamicMemberLookup
public struct MBLSpecificHeapedBaseParams<ModelSpec: MBLModelSpec>
{
    private var params: [PartialKeyPath<ModelSpec.BaseKeys>:MBLHeapedParameter] = [:]
    
    public init(_ spec: ModelSpec, loads: [MBLHeapedParameter]) throws {
        self.params = .init( uniqueKeysWithValues: try loads.map { (try $0.getKeyPath(within: spec), $0) } )
    }
    
    public subscript(dynamicMember keyPath: KeyPath<ModelSpec.BaseKeys, Any>) -> MBLHeapedParameter? {
        return params[keyPath]
    }
}

@dynamicMemberLookup
public struct MBLSpecificHeapedLayerParams<ModelSpec: MBLModelSpec>
{
    private var params: [PartialKeyPath<ModelSpec.LayerKeys>:MBLHeapedParameter] = [:]
    
    public init(_ spec: ModelSpec, loads: [MBLHeapedParameter]) throws {
        self.params = .init( uniqueKeysWithValues: try loads.map { (try $0.getKeyPath(within: spec), $0) } )
    }
    
    public subscript(dynamicMember keyPath: KeyPath<ModelSpec.LayerKeys, Any>) -> MBLHeapedParameter? {
        return params[keyPath]
    }
}

public struct MBLState<ModelSpec: MBLModelSpec> {
    public var spec: ModelSpec
    public var base:            MBLSpecificHeapedBaseParams<ModelSpec>
    public var layers: [UInt32: MBLSpecificHeapedLayerParams<ModelSpec>] = [:]
    
    init(_ spec: ModelSpec,
         heap: MTLHeap,
         loads: [MBLHeapedParameter]) throws {
        self.spec = spec
        
        let baseLoads  = loads.filter { $0.desc.layerNumber == nil }
        let layerLoads = loads.filter { $0.desc.layerNumber != nil }
        
        self.base = try MBLSpecificHeapedBaseParams(spec, loads: baseLoads)
        
        // Layer loads by layer number
        var llbln: [UInt32: [MBLHeapedParameter]] = [:]
        for ll in layerLoads {
            if (llbln[ll.desc.layerNumber!] == nil) {
                llbln[ll.desc.layerNumber!] = []
            }
            llbln[ll.desc.layerNumber!]!.append(ll)
        }
        
        for (ln, ll) in llbln {
            self.layers[ln] = try MBLSpecificHeapedLayerParams(spec, loads: ll)
        }
    }
}

public class MBLParameterStateCollection<THParams, ModelSpec: MBLModelSpec<THParams>> {
    private var modelSpec: ModelSpec
    
    /// Descriptors for the stateDict preparing to be loaded, whose values are gradually filled
    private var stateDictDescriptors:   [(String, MBLStateDictEntryDescriptor?)] = []
    
    public init<TKeys>(_ modelSpec: ModelSpec, specs: [MBLEntrySpec<ModelSpec, TKeys>]) {
        
        self.modelSpec = modelSpec
        
        // Initialize StateDictDescriptors in the same order as the state specs
        specs.forEach { plan in
            self.stateDictDescriptors.append((plan.pythonKey, nil))
        }
    }
    
    public func include<TKeys>(_ desc: MBLStateDictEntryDescriptor, spec: MBLEntrySpec<ModelSpec, TKeys>) throws {
        self.stateDictDescriptors = try self.stateDictDescriptors.map { sd in
            if sd.0 == spec.pythonKey {
                switch desc.layerNumber {
                case .none:
                    _ = try modelSpec.validateShapeToPlan(for: desc.metadata)
                    return (sd.0, desc)
                case .some(let ln):
                    _ = try modelSpec.validateShapeToPlan(for: desc.metadata, layerNumber: ln)
                    return (sd.0, desc)
                }
            } else {
                return sd
            }
        }
    }
    
    public func complete() throws -> [MBLStateDictEntryDescriptor] {
        guard stateDictDescriptors.allSatisfy({$0.1 != nil}) else {
            let missing = stateDictDescriptors.filter({$0.1 == nil }).first
            throw MamBufLoError.incompleteState("MBLParameterStateCollection.next: missing states: \(missing!.0)" )
        }
        
        return stateDictDescriptors.compactMap({$0.1})
    }
    
}

public class MamBufHeapLoader {
    private var device: MTLDevice
    private var heapDescriptor: MTLHeapDescriptor
    private var cmdBuf: MTLCommandBuffer
    private var blitEncoder: MTLBlitCommandEncoder
    private var sizeAndAlignments: [MBLStateDictEntryDescriptor:MTLSizeAndAlign] = [:]
    private var currentNewBufferOffsetInHeap: Int = 0
    
    public var heap: MTLHeap? = nil
    
    public init(device: MTLDevice, cmdQ: MTLCommandQueue) throws {
        self.device = device
        self.heapDescriptor = MTLHeapDescriptor()
        self.heapDescriptor.storageMode = .private
        self.heapDescriptor.hazardTrackingMode = .untracked
        self.heapDescriptor.type = .placement
        self.heapDescriptor.size = 0

        guard let cmdBuf = cmdQ.makeCommandBuffer() else { throw MamBufLoError.failedToMakeCommandBuffer }
        self.cmdBuf = cmdBuf
        self.cmdBuf.label = "StateBuilder_cmdBuf"
        
        guard let blitEncoder = cmdBuf.makeBlitCommandEncoder() else { throw MamBufLoError.failedToMakeCommandEncoder }
        self.blitEncoder = blitEncoder
        self.blitEncoder.label = "StateBuilder_blitEncoder"
    }
    
    public func assignPositionForState(positioning desc: MBLStateDictEntryDescriptor) throws {
        var saa : MTLSizeAndAlign = device.heapBufferSizeAndAlign(length: desc.bytesExpected, options: .storageModePrivate)
        // Apparently, aligning the size so that more resources fit onto heap after this buffer
        saa.size += (saa.size & (saa.align - 1)) + saa.align;
        self.heapDescriptor.size += saa.size
        self.sizeAndAlignments[desc] = saa
    }
    
    public func makeHeap(label: String) throws -> MTLHeap {
        guard let heap = device.makeHeap(descriptor: heapDescriptor) else
        {
            print("Failed to make heap!")
            throw MamBufLoError.failedToMakeMetalHeap
        }
        heap.label = "\(label)_Heap"
        return heap
    }
    
    public func loadToHeap<ModelSpec: MBLModelSpec>(
        _ desc: MBLStateDictEntryDescriptor,
        modelSpec: ModelSpec
    ) throws -> MBLHeapedParameter
    {
        guard let saa = self.sizeAndAlignments[desc] else {
            throw MamBufLoError.incompleteState(desc.prettyName)
        }
        
        if self.heap == nil {
            self.heap = try self.makeHeap(label: modelSpec.name)
        }
        var heap: MTLHeap = self.heap!
        
        print("Loading \(desc.prettyName)...", terminator: "")
        
        // Allocate a temporary buffer containing the data from disk
        var tempBuf = try loadBinaryAsMetalBuffer(binDataPath: desc.dataPath,
                                                  device: device,
                                                  metadata: desc.metadata)
        
        print("OK\t|\tWill allocate", terminator: "...")
        
        // Allocate a new buffer allocated off the heap
        guard let heapBuf = heap.makeBuffer(length: tempBuf.length,
                                            options: .storageModePrivate,
                                            offset: self.currentNewBufferOffsetInHeap) else {
            throw MamBufLoError.failedToMakeMetalHeap
        }
        self.currentNewBufferOffsetInHeap += saa.size
        heapBuf.label = "\(desc.prettyName)_heapBuffer"
        
        self.blitEncoder.copy(from: tempBuf, sourceOffset: 0, to: heapBuf, destinationOffset: 0, size: heapBuf.length)
        tempBuf.contents().deallocate()
        
        print("\(heapBuf.length.formatted(.byteCount(style: .file))) ending at \(self.currentNewBufferOffsetInHeap)")

        
        let shapedBuf = MBLHeapedParameter(data: heapBuf,
                                           desc: desc)
        return shapedBuf
    }
    
    public func commit() throws {
        guard self.heap != nil else { throw MamBufLoError.incompleteState("MamBufHeapLoader.commit: Call makeHeap() and/or loadToHeap() first") }
        blitEncoder.endEncoding()
        cmdBuf.commit()
        print("Loaded model state | Size: \(heap!.currentAllocatedSize.formatted(.byteCount(style: .file)))")
    }
}

public class MamBufLoBuilder<THParams, ModelSpec: MBLModelSpec<THParams>> {
    var nLayers: UInt32
    
    let buildingSpec: ModelSpec
    var hyperparams:  THParams
    
    var baseStatesUnderway:  MBLParameterStateCollection<THParams,ModelSpec>
    var layerStatesUnderway: [UInt32: MBLParameterStateCollection<THParams, ModelSpec>] = [:]
    
    public init(_ modelSpec: ModelSpec, nLayers: UInt32) throws {
        self.nLayers = nLayers
        self.buildingSpec = modelSpec
        self.hyperparams = modelSpec.hp
        
        self.baseStatesUnderway = MBLParameterStateCollection(modelSpec, specs: modelSpec.baseSpecs as! [MBLEntrySpec<ModelSpec, ModelSpec.BaseKeys>])
        
        let perLayerKeys = self.buildingSpec.layerSpecs.map({ $0.pythonKey })
        Array<UInt32>(0..<nLayers).forEach({ n in
            layerStatesUnderway[n] = MBLParameterStateCollection(modelSpec, specs: modelSpec.layerSpecs as! [MBLEntrySpec<ModelSpec, ModelSpec.LayerKeys>])
        })
    }
    
    public func buildStateHeap(device: MTLDevice, cmdQ: MTLCommandQueue) throws -> MBLState<ModelSpec>
    {
        let ascendingLayerStates = layerStatesUnderway.sorted(by: { $0.0 < $1.0 })
        
        let baseStateDescriptors  = try baseStatesUnderway.complete()
        let layerStateDescriptors = try ascendingLayerStates.flatMap { try $0.value.complete() }
        
        var heaper = try MamBufHeapLoader(device: device,
                                          cmdQ: cmdQ)
        
        for desc in (baseStateDescriptors + layerStateDescriptors) {
            try heaper.assignPositionForState(positioning: desc)
        }
        
        var heap = try heaper.makeHeap(label: buildingSpec.name)
        
        var loads: [MBLHeapedParameter] = []
        for desc in (baseStateDescriptors + layerStateDescriptors) {
            loads.append(try heaper.loadToHeap(desc, modelSpec: buildingSpec))
        }
        
        try heaper.commit()
        return try MBLState(buildingSpec, heap: heap, loads: loads)
    }
    
    public func include(_ stateEntryPath: String) throws {
        let desc = try MBLStateDictEntryDescriptor(stateEntryPath)
        switch(desc.layerNumber) {
        case .none:
            let spec = try buildingSpec.findStateDictSpec(for: desc.metadata.stateDictKey)
            try baseStatesUnderway.include(desc, spec: spec)
        case .some(let ln):
            let spec = try buildingSpec.findStateDictSpec(for: desc.metadata.stateDictKey, layerNumber: ln)
            try layerStatesUnderway[ln]!.include(desc, spec: spec)
        }
    }
}
