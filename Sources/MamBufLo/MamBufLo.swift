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
         unknownPreprocessor(String),
         failedToMakeCommandBuffer,
         failedToMakeCommandEncoder,
         failedToMakeMetalBuffer,
         failedToMakeMetalHeap,
         internalError
}

public struct MBLHeapedScratchBuffer {
    public var data: MTLBuffer
    public var shape: [UInt32]
    public var name: String
    
    public lazy var argu: MTLArgumentDescriptor = {
        let argu = MTLArgumentDescriptor()
        
        argu.access = .readWrite
        argu.dataType = .half
        argu.arrayLength = data.length / MemoryLayout<Float16>.stride
        return argu
    }()
}

public struct MBLHeapedParameter {
    public var data: MTLBuffer
    public var desc: MBLStateDictEntryDescriptor
    public lazy var name: String = { desc.prettyName }()
    
    public lazy var argu: MTLArgumentDescriptor = {
        let argu = MTLArgumentDescriptor()
        
        argu.access = .readOnly
        argu.dataType = desc.metadata.baseStride == 2 ? .half : .float
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

public class MamBufHeapLoader {
    private var device: MTLDevice
    private var heapDescriptor: MTLHeapDescriptor
    private var cmdQ: MTLCommandQueue
    private var ppLibrary: MTLLibrary
    private var sizeAndAlignments: [MBLStateDictEntryDescriptor:MTLSizeAndAlign] = [:]
    private var currentNewBufferOffsetInHeap: Int = 0
    
    public var heap: MTLHeap? = nil
    
    public init(device: MTLDevice, cmdQ: MTLCommandQueue, ppLibrary: MTLLibrary) throws {
        self.device = device
        self.heapDescriptor = MTLHeapDescriptor()
        self.heapDescriptor.storageMode = .shared
        // TODO undebug
        //        self.heapDescriptor.storageMode = .private
        self.heapDescriptor.hazardTrackingMode = .untracked
        self.heapDescriptor.type = .placement
        self.heapDescriptor.size = 0

        self.ppLibrary = ppLibrary
        self.cmdQ = cmdQ
    }
    
    public func assignPositionForState(positioning desc: MBLStateDictEntryDescriptor) throws {
        var saa : MTLSizeAndAlign = device.heapBufferSizeAndAlign(length: desc.bytesExpected, options: .storageModeShared) // TODO undebug
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
                
        tempBuf = try modelSpec.preprocess(&tempBuf,
                                           desc: desc,
                                           device: device,
                                           cmdQ: cmdQ,
                                           library: ppLibrary)
        
        print("OK\n\t|\tWill allocate \(tempBuf.length.formatted(.byteCount(style: .file)))", terminator: "...")
        
        // Allocate a new buffer allocated off the heap
        guard let heapBuf = heap.makeBuffer(length: tempBuf.length,
                                            options: .storageModeShared, // TODO undebug
                                            //                                            options: .storageModePrivate,
                                            offset: self.currentNewBufferOffsetInHeap) else {
            throw MamBufLoError.failedToMakeMetalHeap
        }
        self.currentNewBufferOffsetInHeap += saa.size
        heapBuf.label = "\(desc.prettyName)_heapBuffer"
        
        
        guard let cmdBuf = cmdQ.makeCommandBuffer() else { throw MamBufLoError.failedToMakeCommandBuffer }
        cmdBuf.label = "\(desc.prettyName)_StateBuilder_cmdBuf"
        
        guard let blitEncoder = cmdBuf.makeBlitCommandEncoder() else { throw MamBufLoError.failedToMakeCommandEncoder }
        blitEncoder.label = "\(desc.prettyName)_StateBuilder_blitEncoder"
        blitEncoder.copy(from: tempBuf, sourceOffset: 0, to: heapBuf, destinationOffset: 0, size: heapBuf.length)
        
        blitEncoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        tempBuf.contents().deallocate()
        
        print("OK\n\t|\t \(self.currentNewBufferOffsetInHeap): \((Float(currentNewBufferOffsetInHeap) / Float(heap.currentAllocatedSize)).formatted(.percent)) complete")

        
        let shapedBuf = MBLHeapedParameter(data: heapBuf,
                                           desc: desc)
        return shapedBuf
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
    
    public func buildStateHeap(device: MTLDevice, cmdQ: MTLCommandQueue, ppLibrary: MTLLibrary) throws -> MBLState<ModelSpec>
    {
        let ascendingLayerStates = layerStatesUnderway.sorted(by: { $0.0 < $1.0 })
        
        let baseStateDescriptors  = try baseStatesUnderway.complete()
        let layerStateDescriptors = try ascendingLayerStates.flatMap { try $0.value.complete() }
        
        var heaper = try MamBufHeapLoader(device: device,
                                          cmdQ: cmdQ,
                                          ppLibrary: ppLibrary)
        
        for desc in (baseStateDescriptors + layerStateDescriptors) {
            try heaper.assignPositionForState(positioning: desc)
        }
        
        var heap = try heaper.makeHeap(label: buildingSpec.name)
        
        var loads: [MBLHeapedParameter] = []
        for desc in (baseStateDescriptors + layerStateDescriptors) {
            loads.append(try heaper.loadToHeap(desc, modelSpec: buildingSpec))
        }
        
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

func withDebugCapture<T>(on device: MTLDevice, execute closure: () throws -> T) rethrows -> T {
    let sharedCapturer = MTLCaptureManager.shared()
    let customScope = sharedCapturer.makeCaptureScope(device: device)
    customScope.label = "Pls fix it"
    sharedCapturer.defaultCaptureScope = customScope

    let captureDescriptor = MTLCaptureDescriptor()
    captureDescriptor.captureObject = device
    do {
        try sharedCapturer.startCapture(with: captureDescriptor)
    } catch {
        fatalError("Failed to capture: \(error)")
    }
    customScope.begin()

    let result = try closure()

    customScope.end()
    sharedCapturer.stopCapture()

    return result
}
