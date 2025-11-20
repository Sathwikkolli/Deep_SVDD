from utils import load_protocol, build_filename_index, map_protocol_to_paths

def main():
    df = load_protocol()  # reads ../oc_protocol_eval1000.csv
    print("Rows in protocol:", len(df))

    index = build_filename_index()
    print("Indexed files:", sum(len(v) for v in index.values()))

    full_paths = map_protocol_to_paths(df, index)

    # print first few matches to verify
    for i in range(5):
        print(df.iloc[i]["audiofilepath"], "->", full_paths[i])

if __name__ == "__main__":
    main()
