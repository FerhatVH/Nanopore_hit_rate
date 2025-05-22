from read_source.save_in_events import save_in_events
from read_source.npz_reader import get_vector_by_index
from vector_matching.fast_hit_counter import save_in_hits, save_in_hits_average, plot_match_distributions


DEFAULT_DIR = "../data"

def get_max_hit_index(hits_dict):
    if not hits_dict:
        return None
    max_key = max(hits_dict, key=hits_dict.get)
    return int(max_key.split('_')[1])

def main():
    save_in_events(DEFAULT_DIR)
    vector_to_search = get_vector_by_index(npz_path="../temp/2697049_SARS-CoV-31.npz", index=0)
    matching_dict = save_in_hits_average(input_dir=DEFAULT_DIR, vector=vector_to_search)
    file_name, hits_dict, all_dicts = matching_dict
    index = get_max_hit_index(hits_dict)
    file_name = f"../data/{file_name}"
    best_match = get_vector_by_index(npz_path=file_name, index=index)

    plot_match_distributions(all_dicts, save_path="hits_distribution.png")

    print("best_match:")
    print(file_name)
    print(all_dicts)
    print(hits_dict)
    print(get_max_hit_index(hits_dict))


main()