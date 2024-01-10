import Foundation
import Metal

public func loadBinaryAsMetalBuffer(binDataPath: String, device: MTLDevice, metadata: InputMetadata) throws -> MTLBuffer {
    let url = URL(fileURLWithPath: binDataPath)
    let fileHandle = try FileHandle(forReadingFrom: url)
    guard (try? FileManager.default.attributesOfItem(atPath: url.path())) != nil else {
        throw MamBufLoError.invalidFile(url.absoluteString)
    }
    
    var data = try fileHandle.readToEnd()
    defer {
        fileHandle.closeFile()
    }
    
    guard let dataSize = data?.count,
          dataSize == metadata.shape.reduce(MemoryLayout<Float16>.size, {$0 * Int($1)}) else {
        throw MamBufLoError.incompleteLayer(metadata.key)
    }
    var buf: MTLBuffer? = nil
    data!.withUnsafeMutableBytes({ (ptr: UnsafeMutableRawBufferPointer) -> Void in
        buf = device.makeBuffer(bytesNoCopy: ptr.baseAddress!, length: dataSize, options: [.storageModeShared])
    })
    
    guard let someBuf = buf else {
        throw MamBufLoError.failedToMakeMetalBuffer
    }
    return someBuf
}
