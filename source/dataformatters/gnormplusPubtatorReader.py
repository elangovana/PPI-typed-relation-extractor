class GnormplusPubtatorReader:

    def __init__(self):
        pass

    def __call__(self, handle) -> iter:

        for header in handle:

            record = {}
            header_parts = header.split("|")
            record["id"] = header_parts[0]
            record["type"] = header_parts[1]
            record["text"] = header_parts[2].strip("\n")
            annotations = []
            # Loop through annotation
            while True:
                try:
                    annotation_txt = next(handle)
                except StopIteration:
                    return

                if annotation_txt == "\n": break
                # Expected annotation format
                # 19167335        167     170     PTP     Gene    10076
                annotation_parts = annotation_txt.split("\t")
                start_pos, end_pos = annotation_parts[1], annotation_parts[2]
                name = annotation_parts[3]
                type = annotation_parts[4]
                normalised_id = annotation_parts[5].strip("\n")

                annotations.append(
                    {"start": start_pos, "end": end_pos, "name": name, "type": type, "normalised_id": normalised_id})

            record["annotations"] = annotations
            yield record
