# grab unfold fraction for each protein res
all_possible_amino_acids = list('ACDEFGHIKLMNPQRSTVWY')

def grab_uf(xf, seq):
    unfolds = []
    seq = list(seq)
    row_index = 1
    for i in seq:
        x = all_possible_amino_acids.index(i)+1
        FOLD = float(xf.iloc[row_index, x])
        unfolds.append(FOLD)
        row_index+=1
    return unfolds

# grab label for test based on avg unfold fract at spec residue
def grab_label(ef, aa):
    mutation_row = ef[ef['Mutation'] == aa]
    if not mutation_row.empty:
        lab = mutation_row.iloc[0]['EVE_classes_75_pct_retained_ASM']
        try:
            if 'athogenic' in lab:
                return 1
            elif 'enign' in lab:
                return 0
            else:
                return None
        except:
            pass
    else:
        print(f"Mutation {aa} not found in the dataframe.")
        return None

# HBB - grab wt
### Find this file on NEI Commons Website for Protein of Choice
x_path = 'HBB_matrix.csv'
xf = pd.read_csv(x_path)

wt = ''

def extract_fold(xf):
    global wt
    for index, row in xf.iterrows():
        res = str(row.iloc[0])
        res = res[0]
        wt += res


extract_fold(xf)
wt = wt[1:]

# grab HBB unfolding fractions for each residue for wt
unfold_array_hbb = grab_uf(xf, wt)

# initialize data dictionaries
labels_train = {
    "wt-hbb": 0
}
unfold_fractions_train = {
    'wt-hbb': unfold_array_hbb
}

labels_test = {}
unfold_fractions_test = {}

## REPEAT BELOW FOR EACH PROTEIN OF INTEREST
### Download csv file from EVE website
### Download clinvar results for unknown mutations
# Grab mutations from EVE and add into train data
EVE_path = 'HBB_HUMAN.csv'
eve_tdf = pd.read_csv(EVE_path)
eve_tdf['Mutation'] = eve_tdf['wt_aa'].astype(str) + eve_tdf['position'].astype(str) + eve_tdf['mt_aa'].astype(str)

for index, row in eve_tdf.iterrows():
    mut = str(row.iloc[0]) + str(row.iloc[1]) + str(row.iloc[2])
    new_aa = str(row.iloc[2])
    position = int(row.iloc[1])
    new_seq = wt[:position-1] + new_aa + wt[position:]
    label = grab_label(eve_tdf, mut)
    if label is not None:
        labels_train[mut+'-hbb'] = label
        unfold_fractions_train[mut+'-hbb'] = grab_uf(xf, new_seq)

# Start adding test data (UNKNOWN CLINICAL SIGNIFICANCE) to test dict
txt_path = 'clinvar_result_hbb_us.txt'
raw_tdf = pd.read_fwf(txt_path)

tdf = raw_tdf['Name\tGene(s)\tProtein change\tCondition(s)\tAccession\tGRCh37Chromosome\tGRCh37Location\tGRCh38Chromosome\tGRCh38Location\tVariationID\tAlleleID(s)\tdbSNP ID\tCanonical SPDI\tVariant type\tMolecular consequence\tGermline classification\tGermline date last evaluated\tGermline review status\tSomatic clinical impact\tSomatic clinical impact date last evaluated\tSomatic clinical impact review status\tOncogenicity classification\tOncogenicity date last evaluated\tOncogenicity review status']
for i in range(len(tdf)):
    x = tdf[i].split('\t')
    try:
        aa = x[2]
        if ',' in aa:
            raise ValueError("Multiple mutations")
        position = int(aa[1:-1])
        new_aa = aa[-1]
        new_seq = wt[:position-1] + new_aa + wt[position:]
        #sequences[aa+'-hbb'] = new_seq
        label = grab_label(eve_tdf, aa)
        if label is not None:
            labels_test[aa+'-hbb'] = label
            unfold_fractions_test[aa+'-hbb'] = grab_uf(xf, new_seq)
    except ValueError:
        try:
            raw_aa = x[2].split(',')
            for i in raw_aa:
                i = i.strip()
                new_aa = i[-1:]
                old_aa = i[0]
                position = int(i[1:-1])
                aa = old_aa + str(position) + new_aa
                new_seq = wt[:position-1] + new_aa + wt[position:]
                label = grab_label(eve_tdf, aa)
                if label is not None:
                    labels_test[aa+'-hbb'] = label
                    unfold_fractions_test[aa+'-hbb'] = grab_uf(xf, new_seq)
        except IndexError:
            print(f"Index out of bounds: Position {position} is greater than the length of the sequence.")
        except Exception as e:
            print(f"Unexpected error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


# FINAL DATA CURATION
unfold_fractions_train_filtered = {k: v for k, v in unfold_fractions_train.items() if k not in unfold_fractions_test}
labels_train_filtered = {k: v for k, v in labels_train.items() if k not in labels_test}

sequence_data_train = {
    "unfolding_fraction": []
}
for seq_id, uf in unfold_fractions_train_filtered.items():
    sequence_data_train["unfolding_fraction"].append(uf)

df_sequences_train = pd.DataFrame(sequence_data_train)

sequence_data_test = {
    "unfolding_fraction": []
}
for seq_id, uf in unfold_fractions_test.items():
    sequence_data_test["unfolding_fraction"].append(uf)

df_sequences_test = pd.DataFrame(sequence_data_test)

unfolding_fractions_padded = []
for i in df_sequences_train['unfolding_fraction']:
    # change number based on max protein length
    padded = np.zeros(586)
    padded[:len(i)] = i
    unfolding_fractions_padded.append(padded)

unfolding_fractions_padded = []
for i in df_sequences_test['unfolding_fraction']:
    # change number based on max protein length
    padded = np.zeros(586)
    padded[:len(i)] = i
    unfolding_fractions_padded.append(padded)

unfolding_fractions_padded_test = np.array(unfolding_fractions_padded)
unfolding_fractions_padded_train = np.array(unfolding_fractions_padded)

X_train = unfolding_fractions_padded_train
y_train = np.array(list(labels_train_filtered.values()), dtype=np.float32)

X_test = unfolding_fractions_padded_test
y_test = np.array(list(labels_test.values()), dtype=np.float32)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

print(f"Class Weights: {class_weights}")
