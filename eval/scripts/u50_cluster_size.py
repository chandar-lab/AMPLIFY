import xml.etree.ElementTree as ET
import gzip
import os

# Paths
base_path = os.path.join(os.environ.get("DATA_DIR", "/path/to/data"), "uniref/download")
xml_file = os.path.join(base_path, "uniref50.xml.gz")
checkpoint_file = os.path.join(base_path, "checkpoint.txt")
out_prefix = os.path.join(base_path, "uniref100_to_uniref50")

chunk_size = 100_000  # lines per chunk

# Read last checkpoint (last processed UniRef50 cluster ID)
last_cluster = None
if os.path.exists(checkpoint_file):
    with open(checkpoint_file) as ck:
        last_cluster = ck.read().strip()
    print(f"Resuming after cluster {last_cluster}")

# Open XML and stream
ns = "{http://uniprot.org/uniref}"
entry_count = 0
file_index = 0
lines_written = 0

# List existing output files to continue numbering
existing_files = [f for f in os.listdir(base_path) if f.startswith("uniref100_to_uniref50_part") and f.endswith(".csv.gz")]
if existing_files:
    existing_indices = [int(f.split("part")[1].split(".csv.gz")[0]) for f in existing_files]
    file_index = max(existing_indices) + 1

print(f"Starting output file index at {file_index}")
f_out = gzip.open(f"{out_prefix}_part{file_index}.csv.gz", "wt")
f_out.write("uniref100_id,uniref50_rep_id,uniref50_cluster_size\n")

with gzip.open(xml_file, "rt", encoding="ISO-8859-1", errors="replace") as f_in:
    context = ET.iterparse(f_in, events=("end",))

    skip = last_cluster is not None
    for _, elem in context:
        if elem.tag != f"{ns}entry":
            continue

        cluster_id = elem.attrib["id"]

        if skip:
            if cluster_id == last_cluster:
                skip = False
            elem.clear()
            continue

        # Representative member id
        rep_member = elem.find(f"{ns}representativeMember/{ns}dbReference")
        rep_id = rep_member.attrib.get("id") if rep_member is not None else "NA"

        # Cluster size
        prop = elem.find(f"{ns}property[@type='member count']")
        cluster_size = prop.attrib["value"] if prop is not None else "NA"

        # Representative UniRef100 ID
        if rep_member is not None:
            for p in rep_member.findall(f"{ns}property"):
                if p.attrib.get("type") == "UniRef100 ID":
                    f_out.write(f"{p.attrib['value']},{rep_id},{cluster_size}\n")
                    lines_written += 1

        # Members UniRef100 IDs
        for member in elem.findall(f"{ns}member/{ns}dbReference"):
            for p in member.findall(f"{ns}property"):
                if p.attrib.get("type") == "UniRef100 ID":
                    f_out.write(f"{p.attrib['value']},{rep_id},{cluster_size}\n")
                    lines_written += 1

        # Rotate output chunks
        if lines_written >= chunk_size:
            f_out.close()
            file_index += 1
            lines_written = 0
            f_out = gzip.open(f"{out_prefix}_part{file_index}.csv.gz", "wt")
            f_out.write("uniref100_id,uniref50_rep_id,uniref50_cluster_size\n")

        # Update checkpoint
        with open(checkpoint_file, "w") as ck:
            ck.write(cluster_id)

        elem.clear()

f_out.close()
print("Finished successfully.")
