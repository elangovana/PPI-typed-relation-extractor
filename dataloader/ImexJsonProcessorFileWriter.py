import os


class ImexJsonProcessorFileWriter:

    def __init__(self, outdir):
        self.outdir = outdir

    def process(self, imex_file_name, doc_index, doc):
        outfile_name = "{}_{:06d}.json".format(imex_file_name, doc_index)
        out_file = os.path.join(self.outdir, outfile_name)
        with open(out_file, "w") as out_file_handle:
            out_file_handle.write(doc)