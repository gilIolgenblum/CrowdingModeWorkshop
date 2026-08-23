import streamlit as st

def render_sidebar_citation() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '''
        ### Citation

        If you use this app or model in your work, please cite the associated publications and software repository.

        Developed by **Gil I. Olgenblum**

        [ORCID](https://orcid.org/0000-0002-4514-5516) · [GitHub](https://github.com/gilIolgenblum/ProteinCrowding)
        '''
    )

def render_about_and_citation() -> None:
    with st.expander("About, citation, and references", expanded=False):
        st.markdown(
            '''
            ### About

            This app provides an interactive interface for thermodynamic modeling of protein crowding and cosolute effects.

            ### How to cite

            If you use this software or Streamlit app in your work, please cite the associated publications and the software repository:

            [ProteinCrowding GitHub repository](https://github.com/gilIolgenblum/ProteinCrowding)

            Software DOI: to be added after archival release.

            ### Developer

            Developed by **Gil I. Olgenblum**  
            Department of Chemistry, Hebrew University of Jerusalem

            - [ORCID](https://orcid.org/0000-0002-4514-5516)
            - [GitHub repository](https://github.com/gilIolgenblum/ProteinCrowding)

            ### References

            1. Sapir, L.; Harries, D. Macromolecular Stabilization by Excluded Cosolutes: Mean Field Theory of Crowded Solutions. *Journal of Chemical Theory and Computation* **2015**, *11* (7), 3478-3490.
            2. Sapir, L.; Harries, D. Macromolecular Compaction by Mixed Solutions: Bridging versus Depletion Attraction. *Current Opinion in Colloid & Interface Science* **2016**, *22*, 80-87.
            3. Stewart, C. J.; Olgenblum, G. I.; Propst, A.; Harries, D.; Pielak, G. J. Resolving the Enthalpy of Protein Stabilization by Macromolecular Crowding. *Protein Science* **2023**, *32* (3), e4573.
            4. Olgenblum, G. I.; Carmon, N.; Harries, D. Not Always Sticky: Specificity of Protein Stabilization by Sugars Is Conferred by Protein-Water Hydrogen Bonds. *Journal of the American Chemical Society* **2023**, *145* (42), 23308-23320.
            5. Olgenblum, G. I.; Stewart, C. J.; Redvanly, T. W.; Young, O. M.; Lauzier, F.; Hazlett, S.; et al.; Harries, D. Crowding beyond Excluded Volume: A Tale of Two Dimers. *Protein Science* **2025**, *34* (4), e70062.
            6. Redvanly, T. W.; Olgenblum, G. I.; Young, O. M.; Goldenberg, Y.; Stewart, C. J.; Harries, D.; Pielak, G. J. Sugar-Protein Interactions Control Protein-Complex Stability in Crowded Ficoll and Dextran Solutions. *Protein Science* **2026**, *35* (1), e70416.
            '''
        )
