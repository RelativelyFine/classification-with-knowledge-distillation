import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.lines as mlines

def plot_model_architecture_with_activation():
    fig, ax = plt.subplots(figsize=(12, 6))

    # Encoder layers positions
    encoder_layers = [
        {'name': 'Conv1\n3x3, 32, s=2\nReLU', 'x': 0},
        {'name': 'MaxPool\n3x3, s=2', 'x': 1},
        {'name': 'Layer1\nResBlock x2\n32 channels\nReLU', 'x': 2},
        {'name': 'Layer2\nResBlock x2\n64 channels\nReLU', 'x': 3},
        {'name': 'Layer3\nResBlock x3\n128 channels\nReLU', 'x': 4},
        {'name': 'Layer4\nResBlock x2\n256 channels\nReLU', 'x': 5},
    ]

    # Decoder layers positions
    decoder_layers = [
        {'name': 'Deconv1\n4x4, 128, s=2\nReLU', 'x': 4},
        {'name': 'Deconv2\n4x4, 64, s=2\nReLU', 'x': 3},
        {'name': 'Deconv3\n4x4, 32, s=2\nReLU', 'x': 2},
        {'name': 'Deconv4\n4x4, 16, s=2\nReLU', 'x': 1},
        {'name': 'Classifier\n1x1, num_classes', 'x': 0},
    ]

    y_encoder = 3.2
    y_decoder = 2.5

    # Draw encoder layers
    for layer in encoder_layers:
        rect = Rectangle((layer['x'], y_encoder), 0.9, 0.5, edgecolor='black', facecolor='skyblue')
        ax.add_patch(rect)
        ax.text(layer['x'] + 0.45, y_encoder + 0.25, layer['name'], ha='center', va='center', fontsize=8)

    # Draw decoder layers
    for layer in decoder_layers:
        rect = Rectangle((layer['x'], y_decoder), 0.9, 0.5, edgecolor='black', facecolor='lightgreen')
        ax.add_patch(rect)
        ax.text(layer['x'] + 0.45, y_decoder + 0.25, layer['name'], ha='center', va='center', fontsize=8)

    # Draw arrows between encoder layers
    for i in range(len(encoder_layers) - 1):
        ax.annotate("",
                    xy=(encoder_layers[i+1]['x'], y_encoder + 0.25),
                    xytext=(encoder_layers[i]['x'] + 0.9, y_encoder + 0.25),
                    arrowprops=dict(arrowstyle="->"))
        
    # Draw arrow between encoder and decoder
    ax.annotate("",
                xy=(encoder_layers[-1]['x'] + 0.5, y_encoder + 0),
                xytext=(decoder_layers[0]['x'] + 1.5, y_decoder + 0.23),
                arrowprops=dict(arrowstyle="-"))
    ax.annotate("",
                xy=(decoder_layers[0]['x'] + 1.52, y_decoder + 0.25),
                xytext=(decoder_layers[0]['x'] + 0.9, y_decoder + 0.25),
                arrowprops=dict(arrowstyle="<-"))

    # Draw arrows between decoder layers
    for i in range(len(decoder_layers) - 1):
        ax.annotate("",
                    xy=(decoder_layers[i+1]['x'] + 0.9, y_decoder + 0.25),
                    xytext=(decoder_layers[i]['x'], y_decoder + 0.25),
                    arrowprops=dict(arrowstyle="->"))

    # Draw skip connections
    skip_connections = [
        (encoder_layers[2], decoder_layers[2]),
        (encoder_layers[3], decoder_layers[1]),
        (encoder_layers[4], decoder_layers[0]),
    ]
    for enc_layer, dec_layer in skip_connections:
        line = mlines.Line2D([enc_layer['x'] + 0.45, dec_layer['x'] + 0.45],
                             [y_encoder, y_decoder + 0.5],
                             color='red', linestyle='--')
        ax.add_line(line)

    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='skyblue', edgecolor='black', label='Encoder Layer'),
        Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='black', label='Decoder Layer'),
        mlines.Line2D([], [], color='red', linestyle='--', label='Skip Connection'),
        FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", color='black', label='Forward Flow')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Adjust plot
    ax.set_xlim(-0.5, 6)
    ax.set_ylim(2, 4)
    ax.axis('off')

    plt.title('CustomSegmentationModel Architecture with Activations', fontsize=14)
    plt.show()

# Plot the architecture with activation information
plot_model_architecture_with_activation()
