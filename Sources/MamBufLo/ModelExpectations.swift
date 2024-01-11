import Foundation

public struct InputMetadata: Decodable
{
    var key: String
    var shape: [UInt32]
    var stride: [UInt32]
    
    enum CodingKeys : String, CodingKey {
        case key
        case shape
        case stride
    }
    
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        key        = try values.decode(String.self, forKey: .key)
        shape      = try values.decode(Array<UInt32>.self, forKey: .shape)
        stride     = try values.decode(Array<UInt32>.self, forKey: .stride)
    }
}

public protocol ModelStateSpec<T> {
    associatedtype T
    var name: String { get }
    var hp: T { get }
    var stateShapes:      [String:[UInt32]] { get }
    var perLayerStateShapes: [String:[UInt32]] { get }
}

