import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def reduce_image_size(image_path, max_width, max_height):
    """
    Reduce the size of the image to fit within the specified dimensions
    while maintaining the aspect ratio.

    Args:
        image_path (str): Path to the image file.
        max_width (int): Maximum allowed width for the resized image.
        max_height (int): Maximum allowed height for the resized image.

    Returns:
        Image: Resized image object.
    """
    image = Image.open(image_path)
    image.thumbnail((max_width, max_height))  # Resize while maintaining aspect ratio
    return image

def display_page():
    """
    Display the Streamlit page with model training and validation results.
    Includes images, tables, and bar charts to compare WavLM, Wav2Vec2, and Whisper models.
    """
    # Page Title and Description
    st.markdown("<h1 style='text-align: center; color: #4F8EF7;'>Training and Validation Results</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #666;'>Explore the performance of WavLM, Wav2Vec2, and Whisper models during training and validation.</p>",
        unsafe_allow_html=True,
    )

    # Reduce image size while preserving quality
    max_image_width = 500  # Adjust as needed
    max_image_height = 400  # Adjust as needed

    # Display Images with Reduced Size
    st.markdown("### Training Metrics")
    col1, col2 = st.columns(2)

    with col1:
        train_loss_image = reduce_image_size("images/train_loss.jpg", max_image_width, max_image_height)
        st.image(train_loss_image, caption="📉 Train Loss Curve", use_column_width=True)

    with col2:
        train_acc_image = reduce_image_size("images/train_acc.jpg", max_image_width, max_image_height)
        st.image(train_acc_image, caption="📈 Train Accuracy Curve", use_column_width=True)

    st.markdown("### Validation Metrics")
    col3, col4 = st.columns(2)

    with col3:
        val_loss_image = reduce_image_size("images/val_loss.jpg", max_image_width, max_image_height)
        st.image(val_loss_image, caption="📉 Validation Loss Curve", use_column_width=True)

    with col4:
        val_acc_image = reduce_image_size("images/val_acc.jpg", max_image_width, max_image_height)
        st.image(val_acc_image, caption="📈 Validation Accuracy Curve", use_column_width=True)

    # Horizontal Divider
    st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)

    # Table for Insights
    st.markdown("### Insights Table")
    st.markdown(
        """
        <style>
        table {
            margin-left: auto;
            margin-right: auto;
            border-collapse: collapse;
        }
        table th, table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        table th {
            background-color: #f2f2f2;
            color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>WavLM</th>
                    <th>Wav2Vec2</th>
                    <th>Whisper</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Train Loss</td>
                    <td>Lowest, converges quickly</td>
                    <td>Moderate, steady convergence</td>
                    <td>Highest, slower optimization</td>
                </tr>
                <tr>
                    <td>Train Accuracy</td>
                    <td>Highest accuracy achieved</td>
                    <td>Second-best accuracy</td>
                    <td>Lowest accuracy</td>
                </tr>
                <tr>
                    <td>Validation Loss</td>
                    <td>Lowest loss, robust generalization</td>
                    <td>Moderate loss, stable</td>
                    <td>Balanced loss, good generalization</td>
                </tr>
                <tr>
                    <td>Validation Accuracy</td>
                    <td>Stable, slightly behind Whisper</td>
                    <td>Fluctuates early, stabilizes later</td>
                    <td>Highest accuracy, best generalization</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    # Metrics Section
    st.markdown("### Metrics Comparison")

    # Define model metrics
    metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "WavLM": [0.9065, 0.9433, 0.865, 0.9025],
        "Wav2Vec2": [0.8125, 0.9756, 0.6410, 0.7737],
        "Whisper": [0.9140, 0.9304, 0.8950, 0.9123]
    }

    # Plot Metrics as Bar Charts
    categories = metrics["Metric"]
    x = np.arange(len(categories))  # Label locations
    width = 0.25  # Bar width

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 4))  # Increased height for better spacing

    # Create bars for each model
    bars_wavlm = ax.bar(x - width, metrics["WavLM"], width, label="WavLM", color="teal")
    bars_wav2vec2 = ax.bar(x, metrics["Wav2Vec2"], width, label="Wav2Vec2", color="purple")
    bars_whisper = ax.bar(x + width, metrics["Whisper"], width, label="Whisper", color="orange")

    # Add labels and title
    ax.set_ylabel("Scores")
    ax.set_title("Model Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.bbox_inches = "tight"
    # Move the legend further up to avoid interference with bar values
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False)

    # Annotate bars with values on top
    for bars in [bars_wavlm, bars_wav2vec2, bars_whisper]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # Center the text on the bar
                height - 0.1,  # Slightly above the bar
                f"{height:.4f}",  # Format value with four decimals
                ha="center", va="bottom", fontsize=8, color="black"
            )

    # Show the chart
    st.pyplot(fig)

# Call the function to display the page
display_page()
