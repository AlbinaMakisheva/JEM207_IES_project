import streamlit as st
from src.setup import initialize_app
from src.tabs import introduction_tab, analysis_tab
from src.visualization import plot_covid_cases

def main():
    st.sidebar.title("Event-Driven Analysis")
    tab = st.sidebar.radio("Select a Tab", ("Introduction", "Analysis", "COVID-19 Map"))

    # Initialize app and load data
    merged_data, events, covid_data = initialize_app()

    # Render selected tab
    if tab == "Introduction":
        introduction_tab(merged_data, events, covid_data)
    if tab == "Analysis":
        analysis_tab(merged_data, events)
    elif tab == "COVID-19 Map":
        st.write("Displaying the COVID-19 Interactive Map...")
        plot_covid_cases(merged_data)

if __name__ == "__main__":
    main()
