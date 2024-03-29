import XCTest
@testable import MamBufLo

final class BufLoaderTests: XCTestCase {
    var device: MTLDevice!
    var q: MTLCommandQueue!
    var cmdBuf: MTLCommandBuffer!
    var path: String!
    var stateSpec: (any ModelStateSpec<OGHParams>)!
    
    override func setUp() {
        super.setUp()
        guard let device = MTLCreateSystemDefaultDevice(),
              let q = device.makeCommandQueue(),
              let cmdBuf = q.makeCommandBuffer()
        else {
            fatalError("Couldn't do the Metal stuff")
        }
        
        guard let convertedModelsPath = Bundle.module.url(forResource: "converted", withExtension: nil)?.path,
              let availableModels = try? FileManager.default.contentsOfDirectory(atPath: convertedModelsPath).filter({$0 != ".DS_Store"}),
              let model = availableModels.first(where: {$0 == "mamba-130m"}) else {
            fatalError("Couldn't find a model to use. Convert mamba-130m into MamBufLoTests/Resources/converted")
        }
        
        self.device = device
        self.q = q
        self.cmdBuf = cmdBuf
        self.path = convertedModelsPath + "/" + model
        self.stateSpec = OG130()
    }
    
    func testInitialization() throws {
        let MBL = try MamBufLoBuilder<OGHParams>(self.stateSpec, nLayers: self.stateSpec.hp.nLayers)
        XCTAssert(MBL.layerStates.count == self.stateSpec.hp.nLayers)
        XCTAssert(MBL.baseStates.count == self.stateSpec.stateShapes.count)
        XCTAssert(MBL.layerStates[0]?.count == self.stateSpec.perLayerStateShapes.count)
    }
    
    
    func testLoad() throws {
        let contents = try FileManager.default.contentsOfDirectory(atPath: self.path).filter({$0 != ".DS_Store"})
        XCTAssert(contents.count > 0)
        
        var expectedSize = 0
        let MBL = try MamBufLoBuilder(self.stateSpec, nLayers: self.stateSpec.hp.nLayers)
        for cont in contents {
            let metadataPath = self.path + "/" + cont + "/metadata.json"
            let metadataJson = try String(contentsOfFile: metadataPath)
            
            guard let data = metadataJson.data(using: .utf8) else { throw MamBufLoError.invalidFile(metadataPath) }
            let metadata = try JSONDecoder().decode(InputMetadata.self, from: data)
            let binDataPath = self.path + "/" + cont + "/weights.bin"
            
            guard let fileAttributes = try? FileManager.default.attributesOfItem(atPath: binDataPath) else {
                throw MamBufLoError.invalidFile(binDataPath)
            }
            expectedSize += fileAttributes[FileAttributeKey.size] as! Int
            
            try MBL.include(metadata, pathOnDisk: binDataPath)
        }
        var state = try MBL.complete(device: device, cmdQ: q)
        XCTAssertEqual(UInt32(state.layers.count), self.stateSpec.hp.nLayers)
        XCTAssertEqual(state.base.count, self.stateSpec.stateShapes.count)
        XCTAssertEqual(state.layers[0].count, self.stateSpec.perLayerStateShapes.count)
        XCTAssertEqual(state.heartOf.size, state.heartOf.currentAllocatedSize)
        XCTAssertGreaterThanOrEqual(state.heartOf.currentAllocatedSize, expectedSize)
    }
}

struct OG130: ModelStateSpec
{
    let name = "mamba-130m"
    let hp = OGHParams(nLayers: 24,
                       dState: 16,
                       nVocab: 50280,
                       dModel: 768,
                       expand: 2,
                       dConv: 4,
                       dtRank: 768 / 16,
                       dInner: 768 * 2
    )
    var stateShapes: [String : [UInt32]]
    {[
        "embedding.weight":     [hp.nVocab, hp.dModel],
        "lm_head.weight":       [hp.nVocab, hp.dModel],
        "norm_f.weight":        [hp.dModel],
    ]}
    
    var perLayerStateShapes: [String : [UInt32]]
    {[
        "norm.weight":          [hp.dModel],
        "mixer.A_log":          [hp.dInner, hp.dState],
        "mixer.conv1d.bias":    [hp.dInner],
        "mixer.conv1d.weight":  [hp.dInner, 1, hp.dConv],
        "mixer.D":              [hp.dInner],
        "mixer.dt_proj.bias":   [hp.dInner],
        "mixer.dt_proj.weight": [hp.dInner, hp.dtRank],
        "mixer.in_proj.weight": [hp.dInner * hp.expand, hp.dModel],
        "mixer.out_proj.weight":[hp.dModel, hp.dInner],
        "mixer.x_proj.weight":  [hp.dtRank + 2*hp.dState, hp.dInner],
    ]}
}
