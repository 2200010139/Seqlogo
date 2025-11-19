import streamlit as st
import logomaker
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ----------------------------------------------------------
# PAGE TITLE + DESCRIPTION
# ----------------------------------------------------------
st.title("Novel Tool for Sequence Logo Visualization Using PWMs")
st.write("""
### Comparative PWM-based Sequence Logos for **Human** and **Fish**
This tool allows you to upload or enter two PWMs — one for **Human** and one for **Fish** —  
and generates:
- Individual sequence logos  
- Side-by-side comparison  
- Differential (Human vs Fish) enrichment logo  
""")

# ----------------------------------------------------------
# SIDEBAR INPUT
# ----------------------------------------------------------
st.sidebar.header("PWM Input Section")
st.sidebar.write("**Format:** tab-separated values with columns A, C, G, T")

example_pwm = "0.3\t0.2\t0.4\t0.1\n0.1\t0.4\t0.4\t0.1\n0.25\t0.25\t0.25\t0.25"

human_pwm_input = st.sidebar.text_area(
    "Enter HUMAN PWM:",
    value=example_pwm,
    key="human"
)

fish_pwm_input = st.sidebar.text_area(
    "Enter FISH PWM:",
    value=example_pwm,
    key="fish"
)

# ----------------------------------------------------------
# FUNCTION TO PARSE PWM
# ----------------------------------------------------------
def parse_pwm(text):
    df = pd.read_csv(StringIO(text), sep="\t", header=None)
    df.columns = ['A', 'C', 'G', 'T']
    df = df.div(df.sum(axis=1), axis=0)  # Normalize to probabilities
    return df

# ----------------------------------------------------------
# PROCESS INPUT
# ----------------------------------------------------------
try:
    human_pwm = parse_pwm(human_pwm_input)
    fish_pwm = parse_pwm(fish_pwm_input)

    st.subheader("📌 HUMAN PWM")
    st.dataframe(human_pwm)

    st.subheader("📌 FISH PWM")
    st.dataframe(fish_pwm)

    # ------------------------------------------------------
    # PLOT HUMAN LOGO
    # ------------------------------------------------------
    st.subheader("🧬 HUMAN Sequence Logo")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    logo1 = logomaker.Logo(human_pwm, ax=ax1)
    logo1.style_spines(visible=False)
    logo1.style_xticks(rotation=0)
    ax1.set_ylabel("Probability")
    st.pyplot(fig1)

    # ------------------------------------------------------
    # PLOT FISH LOGO
    # ------------------------------------------------------
    st.subheader("🐟 FISH Sequence Logo")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    logo2 = logomaker.Logo(fish_pwm, ax=ax2)
    logo2.style_spines(visible=False)
    logo2.style_xticks(rotation=0)
    ax2.set_ylabel("Probability")
    st.pyplot(fig2)

    # ------------------------------------------------------
    # DIFFERENTIAL LOGO (Human – Fish)
    # ------------------------------------------------------
    st.subheader("🔬 Differential Logo (Human vs Fish)")
    diff_pwm = human_pwm - fish_pwm

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    logo3 = logomaker.Logo(diff_pwm, ax=ax3, color_scheme="classic")
    logo3.style_spines(visible=False)
    logo3.style_xticks(rotation=0)
    ax3.set_ylabel("Human – Fish Difference")

    st.pyplot(fig3)

    # ------------------------------------------------------
    # DOWNLOAD OPTIONS
    # ------------------------------------------------------
    st.sidebar.header("Download Options")

    # PNG - Human Logo
    if st.sidebar.button("Download Human Logo PNG"):
        fig1.savefig("human_logo.png")
        with open("human_logo.png", "rb") as file:
            st.sidebar.download_button(
                label="Download Human Logo",
                data=file,
                file_name="human_logo.png",
                mime="image/png"
            )

    # PNG - Fish Logo
    if st.sidebar.button("Download Fish Logo PNG"):
        fig2.savefig("fish_logo.png")
        with open("fish_logo.png", "rb") as file:
            st.sidebar.download_button(
                label="Download Fish Logo",
                data=file,
                file_name="fish_logo.png",
                mime="image/png"
            )

    # PNG - Differential Logo
    if st.sidebar.button("Download Differential Logo PNG"):
        fig3.savefig("diff_logo.png")
        with open("diff_logo.png", "rb") as file:
            st.sidebar.download_button(
                label="Download Differential Logo",
                data=file,
                file_name="differential_logo.png",
                mime="image/png"
            )

except Exception as e:
    st.error(f"Error processing PWM input: {e}")
