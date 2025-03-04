import shutil
import subprocess
import sys
import tempfile 
import os
import re
import argparse
import logging
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image

TIMEOUT = 10

TEMPLATE = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

def check_dependencies():
    for cmd in ['pdflatex', 'latexmk', 'convert']:
        if not shutil.which(cmd):
            raise RuntimeError(f"Command '{cmd}' not found. Ensure LaTeX and ImageMagick are installed.")

def crop_image(input_path, output_path):
    image = Image.open(input_path)
    image = image.convert('RGB')
    bbox = image.getbbox()
    cropped = image.crop(bbox)
    cropped.save(output_path)


def process_args(args):
    parser = argparse.ArgumentParser(description='Render LaTeX formulas from reference and candidate files.')
    parser.add_argument('--reference-file', dest='reference_file', required=True,
                        help='Path to reference file with ground truth LaTeX formulas.')
    parser.add_argument('--candidate-file', dest='candidate_file', required=True,
                        help='Path to candidate file with predicted LaTeX formulas.')
    parser.add_argument('--output-dir', dest='output_dir', required=True,
                        help='Output directory to store rendered images.')
    parser.add_argument('--replace', dest='replace', action='store_true',
                        help='Replace existing images.')
    parser.add_argument('--num-threads', type=int, default=4,
                        help='Number of rendering threads.')
    parser.add_argument('--log-path', default='render.log',
                        help='Log file path.')
    return parser.parse_args(args)

def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=log_path
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def process_formula(formula):
    formula = formula.strip()
    formula = formula.replace(r'\pmatrix', r'\mypmatrix')
    formula = formula.replace(r'\matrix', r'\mymatrix')
    formula = formula.strip('%')
    
    for space in ["hspace", "vspace"]:
        formula = re.sub(
            rf'{space}{{(.*?)}}',
            lambda m: f'{space}{{{m.group(1).replace(" ", "")}}}',
            formula
        )
    return formula if formula else r'\hspace{1cm}'

def render_formula(formula, output_path, replace):
    if not replace and os.path.exists(output_path):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        base_name = os.path.join(tmpdir, 'temp')
        tex_file = f"{base_name}.tex"
        pdf_file = f"{base_name}.pdf"
        png_file = f"{base_name}.png"

        try:
            with open(tex_file, 'w') as f:
                f.write(TEMPLATE % formula)
            
            # Запуск pdflatex
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_file],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Конвертация в PNG
            subprocess.run(
                ["convert", "-density", "200", "-quality", "100", pdf_file, png_file],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            if os.path.exists(png_file):
                crop_image(png_file, output_path)
                
        except subprocess.CalledProcessError as e:
            logging.error(f"Error processing {output_path}: {str(e)}")
            # Удалите временные файлы при ошибке
            for f in [tex_file, pdf_file, png_file]:
                if os.path.exists(f):
                    os.remove(f)

def main(args):
    check_dependencies()
    params = process_args(args)
    setup_logging(params.log_path)
    
    with open(params.reference_file) as f:
        refs = [process_formula(line) for line in f]
    
    with open(params.candidate_file) as f:
        cands = [process_formula(line) for line in f]
    
    if len(refs) != len(cands):
        logging.error("Files must have equal line count")
        sys.exit(1)

    os.makedirs(os.path.join(params.output_dir, 'images_gold'), exist_ok=True)
    os.makedirs(os.path.join(params.output_dir, 'images_pred'), exist_ok=True)

    tasks = []
    for idx, (ref, cand) in enumerate(zip(refs, cands)):
        gold_path = os.path.join(params.output_dir, 'images_gold', f'img_{idx:04d}.png')
        pred_path = os.path.join(params.output_dir, 'images_pred', f'img_{idx:04d}.png')
        tasks.extend([
            (ref, gold_path, params.replace),
            (cand, pred_path, params.replace)
        ])

    with ThreadPool(params.num_threads) as pool:
        pool.starmap(render_formula, tasks)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info("Rendering complete")