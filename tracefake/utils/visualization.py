import matplotlib.pyplot as plt
import io

def plot_ela_image(original, ela_image):
    """
    Plots the original image and the ELA image side-by-side.
    Returns a matplotlib figure.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(original)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(ela_image)
    ax[1].set_title("Error Level Analysis")
    ax[1].axis('off')
    
    plt.tight_layout()
    return fig

def create_gauge_chart(confidence):
    """
    Placeholder for a more advanced visualization if needed.
    Streamlit has built-in progress bars which are usually sufficient.
    """
    pass
