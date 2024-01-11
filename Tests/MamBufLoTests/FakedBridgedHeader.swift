struct OGHParams {
    // Layer count
    var nLayers: UInt32;
    
    // SSM state expansion factor (aka n)
    var dState: UInt32;
    
    // Vocab size
    var nVocab: UInt32;
    
    // Model dimension (aka d/d_model)
    var dModel: UInt32;
    
    // Block expansion factor
    var expand: UInt32;
    
    // Local convolution width
    var dConv: UInt32;
    
    // Rank of ∆
    var dtRank: UInt32;
    
    // Hidden state dimension (d_in)
    var dInner: UInt32;
};
