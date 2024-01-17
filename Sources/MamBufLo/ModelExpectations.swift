import Foundation
import Metal

public struct MBLEntrySpec<ModelSpec: MBLModelSpec, KeyProvider> {
    public var swiftKey: PartialKeyPath<KeyProvider>
    public var pythonKey: String
    public var shape: [KeyPath<ModelSpec.HParams, UInt32>]
    public var preprocess: String?
    public init(key: PartialKeyPath<KeyProvider>, 
                pythonKey: String,
                shape: [KeyPath<ModelSpec.HParams, UInt32>],
                preprocess: String? = nil) {
        self.swiftKey = key
        self.pythonKey = pythonKey
        self.shape = shape
        self.preprocess = preprocess
    }
}


public protocol MBLModelSpec<HParams> {
    /// A type describing hyperparameters, which define some properties of the model parameters
    associatedtype HParams
    associatedtype BaseKeys
    associatedtype LayerKeys
    associatedtype ScratchKeys
    
    associatedtype BaseSpecs:  Sequence<MBLEntrySpec<Self, BaseKeys>>
    associatedtype LayerSpecs: Sequence<MBLEntrySpec<Self, LayerKeys>>
    associatedtype TypedScratchContext = MBLScratchContext<ScratchKeys> 
    
    var name: String { get }
    var hp: HParams { get }
    
    var baseSpecs:  BaseSpecs  { get }
    var layerSpecs: LayerSpecs { get }
    func scratchSpecs(_ L: UInt32) -> [(keyPath: PartialKeyPath<ScratchKeys>, shape: [UInt32])]
}

extension MBLModelSpec
{
    public func findStateDictSpec(for sdName: String) throws -> MBLEntrySpec<Self, BaseKeys> {
        guard let match = baseSpecs.first(where: { sdName.contains($0.pythonKey) }) else {
            throw MamBufLoError.unknownLayer("MBLModelSpec.findStateShapeEntry: Nothing in spec matches within \(sdName)")
        }
        return match
    }
    
    public func findStateDictSpec(for sdName: String, layerNumber: UInt32? = nil) throws -> MBLEntrySpec<Self, LayerKeys> {
        let specs: Array<MBLEntrySpec<Self, LayerKeys>>? = layerNumber != nil
        ? layerSpecs as? Array<MBLEntrySpec<Self, LayerKeys>>
        : baseSpecs as? Array<MBLEntrySpec<Self, LayerKeys>>
        guard specs != nil else {
            throw MamBufLoError.unknownLayer("Couldn't cast specs!")
        }
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
    
    public func preprocess(_ buffer: inout MTLBuffer, 
                           desc: MBLStateDictEntryDescriptor,
                           device: MTLDevice,
                           cmdQ: MTLCommandQueue,
                           library: MTLLibrary) throws -> MTLBuffer {
        guard let pp = (desc.layerNumber != nil
            ? try findStateDictSpec(for: desc.metadata.stateDictKey, layerNumber: desc.layerNumber).preprocess
            : try findStateDictSpec(for: desc.metadata.stateDictKey).preprocess)
        else {
            return buffer
        }
        
        guard let cmdBuf = cmdQ.makeCommandBuffer() else { throw MamBufLoError.failedToMakeCommandBuffer }
        guard let cmdEnc = cmdBuf.makeComputeCommandEncoder() else { throw MamBufLoError.failedToMakeCommandEncoder }
        guard let ppFn = library.makeFunction(name: pp) else { throw MamBufLoError.unknownPreprocessor(pp) }
        
        print("Pre-processing \(desc.metadata.stateDictKey) with \(pp)...")
        
        cmdEnc.setBuffer(buffer, offset: 0, index: 0)
        let ps = try device.makeComputePipelineState(function: ppFn)
        
        let threadsPerThreadgroup = MTLSizeMake(ps.maxTotalThreadsPerThreadgroup, 1, 1)
        
        let threadsPerGrid = MTLSize(width: buffer.length / desc.metadata.baseStride,
                                     height: 1,
                                     depth: 1)
        cmdEnc.setComputePipelineState(ps)
        cmdEnc.dispatchThreadgroups(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        cmdEnc.endEncoding()
        cmdBuf.commit()
        
        return buffer
    }
    
    public func newScratchContext(device: MTLDevice, L: Int, label: String) throws -> TypedScratchContext {
        let specs = scratchSpecs(UInt32(L))
        var sizeAndAlignments: [PartialKeyPath<ScratchKeys>: MTLSizeAndAlign] = [:]
        var hd = MTLHeapDescriptor()
        hd.storageMode = .shared
        hd.type = .automatic
        hd.size = 0
        
        for spec in specs {
            let shape: [UInt32] = spec.shape
            let bytesExpected = Int(shape.reduce(1, *)) * MemoryLayout<Float16>.stride
            var saa : MTLSizeAndAlign = device.heapBufferSizeAndAlign(length: bytesExpected, options: .storageModeManaged)
            saa.size += (saa.size & (saa.align - 1)) + saa.align;
            hd.size += saa.size
            sizeAndAlignments[spec.keyPath] = saa
        }
        
        guard let heap = device.makeHeap(descriptor: hd) else { throw MamBufLoError.failedToMakeMetalHeap }
        heap.label = "scratchCtx_\(label)/\(L)"
        
        var scratchBufs: [PartialKeyPath<ScratchKeys>:MBLHeapedScratchBuffer] = [:]
        for spec in specs {
            let saa = sizeAndAlignments[spec.keyPath]!
            print("Scratch allocating \(heap.label!) -> \(spec.shape)", terminator: "...")
            guard let heapBuf = heap.makeBuffer(length: saa.size,
                                                options: .storageModeShared) else {
                throw MamBufLoError.failedToMakeMetalHeap
            }
            heapBuf.label = "\(heap.label!).\(spec.keyPath)"
            print("OK: \(heapBuf.label ?? "")")
            scratchBufs[spec.keyPath] = MBLHeapedScratchBuffer(data: heapBuf, shape: spec.shape, name: label)
        }
        
        return MBLScratchContext(L: L, heap: heap, bufs: scratchBufs) as! Self.TypedScratchContext
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
    public var heap: MTLHeap
    public var spec: ModelSpec
    public var base:            MBLSpecificHeapedBaseParams<ModelSpec>
    public var layers: [UInt32: MBLSpecificHeapedLayerParams<ModelSpec>] = [:]
    
    init(_ spec: ModelSpec,
         heap: MTLHeap,
         loads: [MBLHeapedParameter]) throws {
        self.spec = spec
        self.heap = heap
        
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

@dynamicMemberLookup
public struct MBLScratchContext<TScratchKeys> {
    public var L: UInt32
    public var heap: MTLHeap
    private var bufs: [PartialKeyPath<TScratchKeys>:MBLHeapedScratchBuffer] = [:]
    public init(L: Int, heap: MTLHeap, bufs: [PartialKeyPath<TScratchKeys>:MBLHeapedScratchBuffer]) {
        self.L      = UInt32(L)
        self.heap   = heap
        self.bufs   = bufs
    }
    public subscript(dynamicMember dynamicMember: KeyPath<TScratchKeys, Any>) -> MBLHeapedScratchBuffer? {
        return bufs[dynamicMember]
    }
}
