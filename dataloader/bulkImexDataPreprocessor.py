import argparse
import os

from dataloader.imexDataPreprocessor   import ImexDataPreprocessor


def bulk_run(data_dir, out_dir):

    #Get xml files in dir
    for imex_file_name in os.listdir(data_dir):
        if not imex_file_name.endswith(".xml"):
            continue

        # Assuming all xml files are valid imex files.
        full_path = os.path.join(data_dir, imex_file_name)
        print(full_path)
        data_processor = ImexDataPreprocessor()

        with open(full_path, "rb") as xmlhandle:
            i = 1
            for doc in data_processor.run_pipeline(xmlhandle):
                outfile_name = "{}_{:03d}.json".format(imex_file_name, i )
                out_file = os.path.join(out_dir, outfile_name )
                with open(out_file,"w") as out_file_handle:
                    out_file_handle.write(doc)




if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help="The input directory containing the imex files")
    parser.add_argument("out_dir", help="The output dir")

    args = parser.parse_args()
    bulk_run(args.input_dir, args.out_dir)
