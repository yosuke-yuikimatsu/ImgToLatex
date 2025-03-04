import os
import sys
import argparse
import logging
import distance


def process_args(args):
    parser = argparse.ArgumentParser(description='Evaluate text edit distance between two files.')
    
    parser.add_argument('--reference', 
                        type=str, 
                        required=True,
                        help='Path to reference file with ground truth translations')
    
    parser.add_argument('--candidate', 
                        type=str, 
                        required=True,
                        help='Path to candidate file with model translations')
    
    parser.add_argument('--log-path', 
                        dest="log_path",
                        type=str, 
                        default='log.txt',
                        help='Log file path (default: log.txt)')
    
    return parser.parse_args(args)


def main(args):
    parameters = process_args(args)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path
    )
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info('Script started: %s' % __file__)
    
    try:
        with open(parameters.reference) as f:
            ref_lines = [line.strip() for line in f]
        
        with open(parameters.candidate) as f:
            cand_lines = [line.strip() for line in f]
    
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1)
    
    if len(ref_lines) != len(cand_lines):
        logging.error(f"Mismatched number of lines: reference({len(ref_lines)}) vs candidate({len(cand_lines)})")
        sys.exit(1)
    
    total_edit_distance = 0
    total_ref_length = 0
    
    for i, (ref, cand) in enumerate(zip(ref_lines, cand_lines)):
        if i % 100 == 0:
            logging.info(f"Processing line {i}")
        
        ref_tokens = ref.split()
        cand_tokens = cand.split()
        
        max_len = max(len(ref_tokens), len(cand_tokens))
        edit_dist = distance.levenshtein(ref_tokens, cand_tokens)
        
        total_ref_length += max_len
        total_edit_distance += edit_dist
    
    accuracy = 1.0 - float(total_edit_distance)/total_ref_length
    logging.info(f'Total reference length: {total_ref_length}')
    logging.info(f'Total edit distance: {total_edit_distance}')
    logging.info(f'Edit Distance Accuracy: {accuracy:.4f}')

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Execution finished')