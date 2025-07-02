This is my final project for my major in computer science.
In this project my partner Noga and i created an adversarial attack on source separation AI model.
In addition to the attack we tested a simple defense mechanism against the attack with successful results.
for full report please check: final_report file
for simple explanation and results please check the "presentation" file

if you wish to run this you will have to download demucs first: https://github.com/facebookresearch/demucs/blob/main/README.md

afterwards, follow this instructions from the root of our repository:
conda env update -f environment-cpu.yml  # if you don't have GPUs
conda env update -f environment-cuda.yml # if you have GPUs
conda activate demucs
pip install -e .

this should download all dependencies of demucs and our project.

afterwards go to-> demucs main folder->demucs -> apply.py

lines :
316        with th.no_grad():
317            out = model(padded_mix)

change to:
	out = model(padded_mix)






"# How To Run"
process one song:
python attacking scripts/attack_script.py --input file *insert input file here* --save_sources --output_file *output file*


process folder:
python attacking_scripts/process_folder *input folder* 
--script attacking_scripts/attack_script.py

trim and run:
python start.py



attack_results = 
	after attack results,
	separation_prior_attack: separation prior to attack
	separation_after_attack: separation after attack
defended = 
	songs after attacked and after defence
separation_after_attack_and_deface
