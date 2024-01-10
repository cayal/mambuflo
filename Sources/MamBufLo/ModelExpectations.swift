import Foundation

public struct InputMetadata: Decodable
{
    var key: String
    var shape: [Int]
    var stride: [Int]
    
    enum CodingKeys : String, CodingKey {
        case key
        case shape
        case stride
    }
    
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        key        = try values.decode(String.self, forKey: .key)
        shape      = try values.decode(Array<Int>.self, forKey: .shape)
        stride     = try values.decode(Array<Int>.self, forKey: .stride)
    }
}

public protocol ModelStateSpec {
    associatedtype T: MambaHParams
    var name: String { get }
    var nLayers: Int { get }
    var hp: T { get }
    var stateShapes:      [String:[Int]] { get }
    var perLayerStateShapes: [String:[Int]] { get }
}

public protocol MambaHParams
{
    // SSM state expansion factor
    var dState: Int { get }
    
    // Vocab size
    var nVocab: Int { get }
    
    // Model dimensions
    var dModel: Int { get }
    
    // Block expansion factor
    var expand: Int { get }
    
    // Local convolution width
    var dConv: Int { get }
    
    // Rank of ∆
    var dtRank: Int { get }
    
    // Hidden state dimension
    var dInner: Int { get }
    
    init()
}
