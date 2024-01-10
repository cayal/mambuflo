import Metal

enum MamBufLoError: Error {
    case invalidFile(String),
         invalidParameter(String),
         invalidParameterShape(String),
         missingBase(String),
         extraneousLayer(Int),
         missingLayer(Int),
         incompleteLayer(String),
         unknownLayer(String),
         failedToMakeCommandBuffer,
         failedToMakeCommandEncoder,
         failedToMakeMetalBuffer,
         failedToMakeMetalHeap,
         internalError
}

public struct HypotheticalBuf {
    var metadata: InputMetadata
    var pathOnDisk: String
    var byteCount: Int
}

public struct LoadedBuf {
    var data: MTLBuffer
    var meta: InputMetadata
}

public struct MamBufLoSoldier {
    var heartOf: MTLHeap
    var hParams: MambaHParams
    var base: [String:LoadedBuf]
    var layers: [[String:LoadedBuf]]
}

public class MamBufLoBuilder {
    let buildingSpec: any ModelStateSpec
    var hyperparams: MambaHParams
    var baseStates = [String:HypotheticalBuf?]()
    var layerStates = [Int: [String:HypotheticalBuf?]]()
    
    public init(_ modelSpec: any ModelStateSpec) throws {
        self.buildingSpec = modelSpec
        self.hyperparams = modelSpec.hp
        self.baseStates = .init(uniqueKeysWithValues: buildingSpec.stateShapes.map { k, v in (k, nil)} )
        self.layerStates = .init(
            uniqueKeysWithValues: Array(0..<buildingSpec.nLayers).map { n in
                (n, Dictionary<String, HypotheticalBuf?>.init(
                    uniqueKeysWithValues: buildingSpec.perLayerStateShapes.map { k, v in (k, nil) }
                ))
            }
        )
    }
    
    /// Stage 1:
    ///     Validates:
    ///         – That all states have values
    ///         – That there are N monotonically ascending layers
    ///     Calculates:
    ///         – Heap alignment for every state value given its expected size
    ///  Stage 2:
    ///     Creates:
    ///         – Heap with total calculated size
    ///         – Metal buffers aligned within the heap for every state
    ///     Performs:
    ///         – Load of each state path on disk into a temporary MTLBuffer
    ///         – Blit pass copying temporary MTLBuffer contents into heap
    public func complete(device: MTLDevice, cmdQ: MTLCommandQueue) throws -> MamBufLoSoldier
    {
        let ascendingLayerStates = layerStates.sorted(by: { $0.0 < $1.0 })
        var expectedLayerNumbers: [Int] = Array(0..<buildingSpec.nLayers)
        var sizeAndAlignmentsByKey: [String:MTLSizeAndAlign] = [:]
        let descriptor = MTLHeapDescriptor()
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .untracked
        descriptor.type = .placement
        descriptor.size = 0
        
        // Stage 1
        for (k, v) in baseStates {
            guard let v = v else { throw MamBufLoError.missingBase(k) }
            
            var saa : MTLSizeAndAlign = device.heapBufferSizeAndAlign(length: v.byteCount, options: .storageModePrivate)
            // Apparently, aligning the size to that more resources fit onto heap after this buffer
            saa.size += (saa.size & (saa.align - 1)) + saa.align;
            descriptor.size += saa.size
            sizeAndAlignmentsByKey[k] = saa
        }
        
        for (i, kv) in ascendingLayerStates {
            // Expect the layer numbers to be monotonically increasing
            guard expectedLayerNumbers.count > 0 else { throw MamBufLoError.extraneousLayer(i) }
            let iEx = expectedLayerNumbers.removeFirst()
            guard i == iEx else { throw MamBufLoError.missingLayer(iEx) }
            
            for (k, v) in kv {
                guard let v = v else { throw MamBufLoError.incompleteLayer(k) }
                
                var saa : MTLSizeAndAlign = device.heapBufferSizeAndAlign(length: v.byteCount, options: .storageModePrivate)
                // Apparently, aligning the size to that more resources fit onto heap after this buffer
                saa.size += (saa.size & (saa.align - 1)) + saa.align;
                descriptor.size += saa.size
                sizeAndAlignmentsByKey[k] = saa
            }
        }
        
        // Expect there to be exactly N layers
        guard expectedLayerNumbers.count == 0 else {
            throw MamBufLoError.missingLayer(expectedLayerNumbers.removeFirst())
        }
        
        // Stage 2
        guard let heap = device.makeHeap(descriptor: descriptor) else
        {
            print("Failed to make heap!")
            throw MamBufLoError.failedToMakeMetalHeap
        }
        
        heap.label = "\(buildingSpec.name)_Heap"
        var soldier = MamBufLoSoldier(heartOf: heap, hParams: hyperparams, base: [:], layers: [])
        
        guard let cmdBuf = cmdQ.makeCommandBuffer() else { throw MamBufLoError.failedToMakeCommandBuffer }
        cmdBuf.label = "StateBuilder_cmdBuf"
        
        guard let blitEncoder = cmdBuf.makeBlitCommandEncoder() else { throw MamBufLoError.failedToMakeCommandEncoder }
        blitEncoder.label = "StateBuilder_blitEncoder"
        
        var currentOffset = 0
        for (k, v) in baseStates {
            guard let v = v else { throw MamBufLoError.incompleteLayer(k) }

            var tempBuf = try loadBinaryAsMetalBuffer(binDataPath: v.pathOnDisk, device: device, metadata: v.metadata)
            
            guard let heapBuf = heap.makeBuffer(length: tempBuf.length, options: .storageModePrivate, offset: currentOffset)
            else { throw MamBufLoError.failedToMakeMetalBuffer }
            
            guard sizeAndAlignmentsByKey[k] != nil else { throw MamBufLoError.internalError }
            currentOffset += sizeAndAlignmentsByKey[k]!.size
            heapBuf.label = "\(k)_heapBuffer"
            blitEncoder.copy(from: tempBuf, sourceOffset: 0, to: heapBuf, destinationOffset: 0, size: heapBuf.length)
            tempBuf = heapBuf
            soldier.base[k] = LoadedBuf(data: heapBuf, meta: v.metadata)
        }
        
        
        for (i, kv) in ascendingLayerStates {
            soldier.layers.append([:])
            for (k, v) in kv {
                print("Loading: \(k)", terminator: "...")
                var tempBuf = try loadBinaryAsMetalBuffer(binDataPath: v!.pathOnDisk, device: device, metadata: v!.metadata)
                print("OK\t|\tWill allocate", terminator: "...")
                
                guard let heapBuf = heap.makeBuffer(length: tempBuf.length, options: .storageModePrivate, offset: currentOffset)
                else { throw MamBufLoError.failedToMakeMetalBuffer }
                
                guard sizeAndAlignmentsByKey[k] != nil else { throw MamBufLoError.internalError }
                currentOffset += sizeAndAlignmentsByKey[k]!.size
                heapBuf.label = "layer_\(i)_\(k)_heapBuffer"
                blitEncoder.copy(from: tempBuf, sourceOffset: 0, to: heapBuf, destinationOffset: 0, size: heapBuf.length)
                tempBuf = heapBuf
                soldier.layers[i][k] = LoadedBuf(data: heapBuf, meta: v!.metadata)
                print("\(heapBuf.length.formatted(.byteCount(style: .file))) ending at \(currentOffset)")
            }
        }
        
        blitEncoder.endEncoding()
        cmdBuf.commit()
        print("Loaded model state | Size: \(heap.currentAllocatedSize.formatted(.byteCount(style: .file)))")
        return soldier
    }
    
    public func include(_ metadata: InputMetadata, pathOnDisk: String) throws {
        switch(try /layers\.([0-9]+)/.firstMatch(in: metadata.key)) {
        case .none:
            try includeOuterState(metadata, pathOnDisk: pathOnDisk)
        case .some(let match):
            guard let layerNumber = Int(match.output.1) else { throw MamBufLoError.unknownLayer(String(match.output.1)) }
            try includeLayerState(layerNumber, metadata, pathOnDisk: pathOnDisk)
        }
    }
    
    public func matchToPlanElement(_ plan: Dictionary<String, [Int]>, _ metadata: InputMetadata) throws -> Dictionary<String, [Int]>.Element {
        guard let planElement = plan.first(where: {k, v in metadata.key.contains(k)}) else {
            throw MamBufLoError.invalidParameter("\(metadata.key) has no parts matching: \(plan.keys.debugDescription)")
        }
        guard metadata.shape == planElement.value else {
            throw MamBufLoError.invalidParameterShape("\(metadata.key): \(metadata.shape.debugDescription) is not according to plan \(planElement.key): \(planElement.value)")
        }
        return planElement
    }
    
    public func includeOuterState(_ metadata: InputMetadata, pathOnDisk: String) throws {
        let planElement = try matchToPlanElement(buildingSpec.stateShapes, metadata)
        baseStates[planElement.key] = HypotheticalBuf(metadata:metadata,
                                                      pathOnDisk: pathOnDisk,
                                                      byteCount: metadata.shape.reduce(1, *) * MemoryLayout<Float16>.stride)
    }
    
    public func includeLayerState(_ layerNumber: Int, _ metadata: InputMetadata, pathOnDisk: String) throws {
        guard layerStates[layerNumber] != nil else { throw MamBufLoError.unknownLayer(String(layerNumber)) }
        let planElement = try matchToPlanElement(buildingSpec.perLayerStateShapes, metadata)
        layerStates[layerNumber]![planElement.key] = HypotheticalBuf(metadata: metadata,
                                                                     pathOnDisk: pathOnDisk,
                                                                     byteCount: metadata.shape.reduce(1, *) * MemoryLayout<Float16>.stride)
    }
}
