# import os
# import json

# def merge_conversations(folder_path, output_file):
#     merged_conversations = []
    
#     # Loop over each file in the folder
#     for filename in os.listdir(folder_path):
#         print(f'{filename = }')
#         if filename.endswith('.jsonl'):
#             file_path = os.path.join(folder_path, filename)
#             print(f'{file_path=}')
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as file:
#                     data = json.load(file)
#                     if 'conversations' in data and isinstance(data['conversations'], list):
#                         num_entries = len(data['conversations'])
#                         print(f"{filename}: found {num_entries} conversation entries")
#                         merged_conversations.extend(data['conversations'])
#                     else:
#                         print(f"{filename}: 'conversations' key not found or is not a list")
#             except json.JSONDecodeError as e:
#                 print(f"{filename}: JSON decode error - {e}")
#             except Exception as e:
#                 print(f"{filename}: Error processing file - {e}")
    
#     # Write the merged conversations to the output file
#     merged_data = {"conversations": merged_conversations}
#     with open(output_file, 'w', encoding='utf-8') as outfile:
#         json.dump(merged_data, outfile, indent=4)
    
#     print(f"Merged {len(merged_conversations)} conversation entries into {output_file}")

# # Example usage:
# if __name__ == '__main__':
#     folder_path = "/Users/akashbhansali/Documents/assignments/llm/final_projec/llm-UTNchatbot/data/q_a/Q_A/"  # Replace with the path to your folder
#     output_file = "UTN_q_a_merged.json"              # The output file name
#     merge_conversations(folder_path, output_file)

import os
import json

def merge_jsonl_files(input_folder, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_folder):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)

if __name__ == "__main__":
    input_folder = "/Users/akashbhansali/Documents/assignments/llm/final_projec/llm-UTNchatbot/data/q_a/sig_Q_A"  # Folder containing JSONL files
    output_file = "data/q_a/sig_q_a_merged.jsonl"  # Output merged file
    merge_jsonl_files(input_folder, output_file)
    print(f"Merged files into {output_file}")
