"""
@Thanh Truc Bui
"""
import collections
import os
from dotenv import load_dotenv

load_dotenv() # load environment variables

abs_path_to_split30_fasta = os.getenv("FASTA_ENZYMES","not found")
abs_path_to_non_enzyme_fasta = os.getenv("FASTA_NON_ENZYMES","not found")

def read_csv(path):  #read the protein mass table
    masses={}
    with open(path, "r") as file:
        for line in file:
            info = line.split("\t")
            char = info[0]
            mass = info[1]
            masses[char] = mass
    return masses

protein_masses=read_csv(r'/home/trucbui/PBL/Dataset/Dataset/mass.tsv')


def readfasta(fasta, is_enzyme):
    seq_list=[]
    a=0
    if is_enzyme==True:
        a=1
    with open(fasta, "r") as path:
        seq = ""
        header = path.readline().rstrip()[1:]
        for line in path.readlines():
            if line.startswith(">"):
                seq_list.append((a,header, seq))
                seq = ""
                header = line.rstrip()[1:]
            else:
                seq += line.rstrip()
        seq_list.append((a,header, seq))

    seq_list2=[]

    for (a,header, seq) in seq_list:
        mass = 0.0
        for char in seq:
            if char in protein_masses.keys():
                mass += float(protein_masses[char])
                mass = round(mass, 3)
        seq_list2.append((a, header, seq, mass))
    return seq_list2

enzyme_seqs=readfasta(abs_path_to_split30_fasta,True)
enzyme_only_seqs=[]
for (a,header,seq,mass) in enzyme_seqs:
    enzyme_only_seqs.append(seq)
non_enzyme_seqs=readfasta(abs_path_to_non_enzyme_fasta, False)
non_enzyme_only_seqs=[]
for (a,header,seq,mass) in non_enzyme_seqs:
    non_enzyme_only_seqs.append(seq)

# Finding a shared motif
def find_motif(seq_list):
    sort_seqs=sorted(seq_list,key=len)
    shortest=sort_seqs[0]
    motif=''
    for i in range(len(shortest)):
        for j in range(i,len(shortest)):
            m = shortest[i:j+1]
            found = True
            for seq in seq_list[1:]:
                if not m in seq:
                    found = False
                    break
            if found==True and len(m)>len(motif):
                motif=m
    return motif

"""
ok this is not really promising I guess
"""
#print(find_motif(enzyme_only_seqs))
#print(find_motif(non_enzyme_only_seqs))

# Find out the most common last amino acid of the enzymes and non-enzymes set
def most_common_last_aa(seq_list):
    last_aa_list=dict()
    for seq in seq_list:
        last_aa=seq[-1:]
        if not last_aa in last_aa_list:
            last_aa_list[last_aa] = 0
        last_aa_list[last_aa]+=1

    max=0
    common_aa=[]
    for aa in last_aa_list.keys():
        if not aa=="X":
            if last_aa_list[aa]>max:
                common_aa.clear()
                common_aa.append(aa)
            if last_aa_list[aa]==max:
                common_aa.append(aa)
    return common_aa

enzymes_common_last_aa=most_common_last_aa(enzyme_only_seqs)
non_enzymes_common_last_aa=most_common_last_aa(non_enzyme_only_seqs)
print(enzymes_common_last_aa)
print(non_enzymes_common_last_aa)


def writecsv(seqlist1, seqlist2, csv_path):
    with open(csv_path, "w") as out:
        out.write("Label" + "\t" + "Header" + "\t" + "Sequence" +"\t" + "Sequence length" + "\t" + "Mass" + "\t" +
                  "Most frequent aa" + "\t" + "Last aa is " +enzymes_common_last_aa[0] +
                  "\t" + "Last aa is " +non_enzymes_common_last_aa[0]+ "\n")
        for i in range(4):
            for (a,header,seq,mass) in seqlist1:
                out.write(str(a)+"\t"+ header+ "\t" +seq + "\t"+ str(len(seq))+ "\t" + str(mass) +"\t" +
                          collections.Counter(seq).most_common(1)[0][0] + "\t" + str(int(seq[-1:]==enzymes_common_last_aa[0])) +
                          "\t" + str(int(seq[-1:]==non_enzymes_common_last_aa[0])) +"\n")
        for (a,header,seq,mass) in seqlist2:
            out.write(str(a) + "\t" + header + "\t" + seq + "\t" + str(len(seq)) + "\t" + str(mass) + "\t" +
                      collections.Counter(seq).most_common(1)[0][0] + "\t" + str(int(seq[-1:] == enzymes_common_last_aa[0])) +
                      "\t" + str(int(seq[-1:] == non_enzymes_common_last_aa[0])) + "\n")
        out.close()


writecsv(enzyme_seqs, non_enzyme_seqs , r"/home/trucbui/PBLGit/random_forest/binary_classification_train.csv")
