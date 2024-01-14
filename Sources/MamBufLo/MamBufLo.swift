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

public class MBLStateDictKeyMap<ModelSpec, TKey, HParams>
where ModelSpec: MBLModelSpec<HParams>, TKey: Sequence<KeyPath<HParams, UInt32>> {
    public var name: String
    public var kp: KeyPath<ModelSpec, TKey>
    public init(name: String, kp: KeyPath<ModelSpec, TKey>) {
        self.name = name
        self.kp = kp
    }
}

public protocol MBLBaseStateKeys: RawRepresentable {
    var rawValue: String { get }
}

public protocol MBLModelSpec<HParams> {
    /// A type describing hyperparameters, which define some properties of the model parameters
    associatedtype HParams
    associatedtype BaseStateShapeSpec:  Sequence<KeyPath<HParams, UInt32>>
    associatedtype LayerStateShapeSpec: Sequence<KeyPath<HParams, UInt32>>
    associatedtype BaseStateDictKeyMap:  MBLStateDictKeyMap<Self, BaseStateShapeSpec, HParams>
    associatedtype LayerStateDictKeyMap: MBLStateDictKeyMap<Self, LayerStateShapeSpec, HParams>
    
    /// The name of a model folder on disk
    var name: String { get }
    var hp: HParams { get }
    
    /// A dictionary mapping:
    /// Strings, which may match as a substring of a state-dict key, to:
    /// Keypaths within this model, which define the shape we expect that state-dict entry to conform to, expressed as the hyperparameter members.
    var baseStateShapes:     [BaseStateDictKeyMap ]  { get }
    var perLayerStateShapes: [LayerStateDictKeyMap] { get }
}

extension MBLModelSpec
{
    public func findStateDictKeyMap(for sdName: String) throws
    -> BaseStateDictKeyMap {
        guard let match = baseStateShapes.first(where: { sdName.contains($0.name) }) else {
            throw MamBufLoError.unknownLayer("MBLModelSpec.findStateShapeEntry: Nothing in spec matches within \(sdName)")
        }
        return match
    }
    
    public func findStateDictKeyMap(for sdName: String, layerNumber: UInt32) throws
    -> LayerStateDictKeyMap {
        guard let match = perLayerStateShapes.first(where: { sdName.contains($0.name) }) else {
            throw MamBufLoError.unknownLayer("MBLModelSpec.findStateShapeEntry: Nothing in spec matches within \(sdName)")
        }
        return match
    }
    
    public func validateShapeToPlan(for metadata: MBLStateDictMetadata, layerNumber: UInt32?) throws -> [UInt32] {
        let match = layerNumber != nil
                    ? try findStateDictKeyMap(for: metadata.stateDictKey, layerNumber: layerNumber!) as! MBLStateDictKeyMap
                    : try findStateDictKeyMap(for: metadata.stateDictKey)
        let plannedShape = self[keyPath: match.kp].map({ hp[keyPath: $0] })
        guard metadata.shape == plannedShape else {
            let actual   = "\(metadata.stateDictKey): \(metadata.shape.debugDescription)"
            let expected = "\(match.kp): \(plannedShape)"
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
    -> KeyPath<ModelSpec, ModelSpec.BaseStateShapeSpec> {
        return try modelSpec.findStateDictKeyMap(for: desc.metadata.stateDictKey).kp
    }
    
    public func getKeyPath<ModelSpec: MBLModelSpec>(within modelSpec: ModelSpec) throws
    -> KeyPath<ModelSpec, ModelSpec.LayerStateShapeSpec> {
        return try modelSpec.findStateDictKeyMap(for: desc.metadata.stateDictKey, layerNumber: desc.layerNumber!).kp 
    }
}


@dynamicMemberLookup
public struct MBLSpecificHeapedBaseParams<ModelSpec: MBLModelSpec>
{
    private var params: [KeyPath<ModelSpec, ModelSpec.BaseStateShapeSpec>:MBLHeapedParameter] = [:]
    
    public init(_ spec: ModelSpec, loads: [MBLHeapedParameter]) throws {
        self.params = .init( uniqueKeysWithValues: try loads.map { (try $0.getKeyPath(within: spec), $0) } )
    }
    
    public subscript(dynamicMember keyPath: KeyPath<ModelSpec, ModelSpec.BaseStateShapeSpec>) -> MBLHeapedParameter? {
        return params[keyPath]
    }
}

@dynamicMemberLookup
public struct MBLSpecificHeapedLayerParams<ModelSpec: MBLModelSpec>
{
    private var params: [KeyPath<ModelSpec, ModelSpec.LayerStateShapeSpec>:MBLHeapedParameter] = [:]
    
    public init(_ spec: ModelSpec, loads: [MBLHeapedParameter]) throws {
        self.params = .init( uniqueKeysWithValues: try loads.map { (try $0.getKeyPath(within: spec), $0) } )
    }
    
    public subscript(dynamicMember keyPath: KeyPath<ModelSpec, ModelSpec.LayerStateShapeSpec>) -> MBLHeapedParameter? {
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
    private var forLayers: Bool
    
    /// The description from the modelSpec, with the shape expectations filled in
    private var stateShapeExpectations: [String: [UInt32]] = [:]
    
    /// Descriptors for the stateDict preparing to be loaded, whose values are gradually filled
    private var stateDictDescriptors:   [String: MBLStateDictEntryDescriptor?] = [:]
    
    public init(_ modelSpec: ModelSpec, forLayers: Bool) {
        self.modelSpec = modelSpec
        self.forLayers = forLayers
        let shapeSpecs = self.forLayers 
        ? modelSpec.perLayerStateShapes as! any Sequence<MBLStateDictKeyMap<ModelSpec, [KeyPath<THParams, UInt32>], THParams>>
        : modelSpec.baseStateShapes as! any Sequence<MBLStateDictKeyMap>
        
        self.stateShapeExpectations = .init(uniqueKeysWithValues: shapeSpecs.map({ entry in
            (key: entry.name, value: modelSpec[keyPath: entry.kp].map { modelSpec.hp[keyPath: $0]})
        }))
        
        let planKeys = self.stateShapeExpectations.keys
        planKeys.forEach({ k in self.stateDictDescriptors[k] = nil })
    }
    
    public func include(_ desc: MBLStateDictEntryDescriptor) throws {
        guard let key = self.stateShapeExpectations.keys.first(where: {desc.metadata.stateDictKey.contains($0)}) else {
            throw MamBufLoError.unknownLayer("Did not match \(desc.metadata.stateDictKey)" +
                                             "to any of \(stateShapeExpectations.keys)")
        }
        
        guard self.stateDictDescriptors[key] == nil else {
            throw MamBufLoError.extraneousStateEntry(desc.metadata.stateDictKey)
        }
        
        let shape = try modelSpec.validateShapeToPlan(for: desc.metadata, layerNumber: desc.layerNumber)
        self.stateDictDescriptors[key] = desc
    }
    
    public func complete() throws -> [MBLStateDictEntryDescriptor] {
        guard stateDictDescriptors.allSatisfy({$0.value != nil}) else {
            let missing = stateDictDescriptors.filter({$0.value == nil }).keys
            throw MamBufLoError.incompleteState("MBLParameterStateCollection.next: missing states: \(missing)" )
        }
        
        return stateDictDescriptors.compactMap({$0.value})
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
        
        self.baseStatesUnderway = MBLParameterStateCollection(modelSpec, forLayers: false)
        
        let perLayerKeys = self.buildingSpec.perLayerStateShapes.map({ $0.name })
        Array<UInt32>(0..<nLayers).forEach({ n in
            layerStatesUnderway[n] = MBLParameterStateCollection(modelSpec, forLayers: true)
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
        let exp = try MBLStateDictEntryDescriptor(stateEntryPath)
        let forLayers = exp.layerNumber != nil
        switch(exp.layerNumber) {
        case .none:
            let kp = try buildingSpec.findStateDictKeyMap(for: exp.metadata.stateDictKey)
            try baseStatesUnderway.include(exp)
        case .some(let ln):
            let kp = try buildingSpec.findStateDictKeyMap(for: exp.metadata.stateDictKey, layerNumber: ln)
            try layerStatesUnderway[ln]!.include(exp)
        }
    }
}
