import Foundation
import Metal
#if arch(i386) || arch(x86_64)
   @available(macOS 14.0, iOS 17.0, *)
   typealias Float16 = Float32
#endif

public func loadBinaryAsMetalBuffer(binDataPath: String, device: MTLDevice, metadata: MBLStateDictMetadata) throws -> MTLBuffer {
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
          dataSize == metadata.shape.reduce(metadata.baseStride, {$0 * Int($1)}) else {
        throw MamBufLoError.incompleteLayer(metadata.stateDictKey)
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
