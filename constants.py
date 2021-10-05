import os

class Constants():
    # Chord length 
    BEAT_PER_CHORD = 2
    CHORDS_PER_BAR = 2

    # 96 simplified chords
    NUM_CHORDS = 96
    BARS_PER_CONDUCTOR = 8

    # Beat resolution
    BEAT_RESOLUTION = 24
    
    # Max chord sequence
    MAX_SEQUENCE_LENGTH = 272
    
    
class CVAE_Constants():
    # Model parameters
    ENCODER_HIDDEN_SIZE = 256 
    DECODER_HIDDEN_SIZE = 256 
    
    LATENT_SIZE = 16 
    
    ENCODER_NUM_LAYERS = 2 
    DECODER_NUM_LAYERS = 2 
    
    PRENET_SIZE = 256
    PRENET_NUM_LAYERS = 2 
    
    NUM_CHORDS = 96
    
class CVAE_All_Chords_Constants():
    # Model parameters
    ENCODER_HIDDEN_SIZE = 256 * 2 * 2
    DECODER_HIDDEN_SIZE = 256 * 2 * 2
    
    LATENT_SIZE = 16 * 4
    
    ENCODER_NUM_LAYERS = 2 
    DECODER_NUM_LAYERS = 2 
    
    PRENET_SIZE = 256
    PRENET_NUM_LAYERS = 2 
    
    NUM_CHORDS = 633

class MusicVAE_Constants():
    
    # Model parameters
    ENCODER_HIDDEN_SIZE = 1024
    CONDUCTOR_HIDDEN_SIZE = 1024
    DECODER_HIDDEN_SIZE = 1024
    PRENET_HIDDEN_SIZE = 1024
    
    LATENT_SIZE = 256
    
    ENCODER_NUM_LAYER = 2
    CONDUCTOR_NUM_LAYER = 2
    DECODER_NUM_LAYER = 2
    PRENET_NUM_LAYER = 2
    
    # Training parameters
    TEACHER_FORCING = True
