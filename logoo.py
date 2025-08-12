import streamlit as st
import logomaker
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit app title and description
st.title("Sequence Logo Generator from PWM")
st.write("""
Upload or enter a Position Weight Matrix (PWM) to generate a sequence logo.
The PWM should have nucleotides as columns (A, C, G, T) and positions as rows.
""")

# Sidebar for PWM input
st.sidebar.header("Input PWM")
st.sidebar.write("Example format (tab-separated or CSV):")
st.sidebar.code("A\tC\tG\tT\n0.2\t0.3\t0.4\t0.1\n0.1\t0.4\t0.4\t0.1\n...")

# Text area for PWM input
pwm_input = st.sidebar.text_area(
    "Enter PWM (one row per position)",
    value="0.2\t0.3\t0.4\t0.1\n0.1\t0.4\t0.4\t0.1\n0.25\t0.25\t0.25\t0.25"
)

# Parse the PWM into a DataFrame
try:
    from io import StringIO
    pwm_df = pd.read_csv(StringIO(pwm_input), sep="\t", header=None)
    
    # Assign column names for nucleotides
    pwm_df.columns = ['A', 'C', 'G', 'T']
    
    # Normalize rows if not already probabilities
    pwm_df = pwm_df.div(pwm_df.sum(axis=1), axis=0)

    # Display the PWM
    st.subheader("Position Weight Matrix")
    st.dataframe(pwm_df)

    # Generate the sequence logo
    st.subheader("Generated Sequence Logo")
    fig, ax = plt.subplots(figsize=(10, 5))
    logo = logomaker.Logo(pwm_df, ax=ax)
    logo.style_spines(visible=False)
    logo.style_xticks(rotation=0, fmt='%d', anchor=0)
    logo.ax.set_ylabel("Probability")

    st.pyplot(fig)

    # Download option
    st.sidebar.subheader("Download Options")
    if st.sidebar.button('Download Logo as PNG'):
        fig.savefig('sequence_logo.png')
        with open("sequence_logo.png", "rb") as file:
            st.sidebar.download_button(
                label="Download Logo",
                data=file,
                file_name="sequence_logo.png",
                mime="image/png"
            )

except Exception as e:
    st.error(f"Error processing PWM: {e}")
