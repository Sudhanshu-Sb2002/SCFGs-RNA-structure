# SCFGs_RNA
Using Stochastic context free grammars to predict RNA secondary structure.Mostly following the approach outlined in the paper https://doi.org/10.1093/bioinformatics/15.6.446

![RNA stucture](tmp_data_files/Predicted vs actual.png?raw=true)
To run the code, run src/main.py.
There is general CFG class that can be used to create a context free grammar with 4 types of rules: Emissions, Transitions, Replacements, Emission-replacements
 
This project uses python, and some standard python libraries. Numba has been used to speed up the code by JITting, and is required to run the code.

The dataset used is linked at https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-340 .The folder tmp_data_files has somme big files that are not inlcuded in the repo (mainly temporary files that are used to speed up the code). These files can be downloaded from the link: https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/sudhanshub_iisc_ac_in/EkN46R3n2atCn0mktg2SE7sBFWhg3rrRUoKkgOTzNWSNWA?e=2HHtDe


