use p3_blake2s::{Blake2s256, T8Blake2s};
use p3_blake3::{Blake3, T8Blake3};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{MerkleTree, MerkleTreeMmcs};
use p3_sha256::{Sha256Compress, T8Sha256, T253Sha256};
use p3_symmetric::{CompressionFunctionFromHasher, CryptographicHasher, PseudoCompressionFunction};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// T8 records of one, two and three stages: `32 + 224 k` bytes.
const T8_LENS: [usize; 3] = [256, 480, 704];

/// T253 records of one, two and three stages: `32 + 221 k` bytes.
const T253_LENS: [usize; 3] = [253, 474, 695];

/// A matrix of `rows` random records of `len` bytes.
fn records(rows: usize, len: usize, seed: u64) -> RowMajorMatrix<u8> {
    let mut rng = SmallRng::seed_from_u64(seed);
    RowMajorMatrix::new((0..rows * len).map(|_| rng.random()).collect(), len)
}

/// Plonky3's tree over T8 leaves has the textbook root of the same leaves and nodes.
fn textbook_root<H, C>(leaf: &H, node: &C, lens: [usize; 3])
where
    H: CryptographicHasher<u8, [u8; 32]> + Sync,
    C: PseudoCompressionFunction<[u8; 32], 2> + Sync,
{
    // Invariant: only the leaves change; the tree is the ordinary binary tree over them.
    //
    //     v_0,i = T8(X_i)
    //     v_l+1,j = C(v_l,2j, v_l,2j+1)
    for (rows, len) in [
        (1, lens[0]),
        (2, lens[0]),
        (64, lens[0]),
        (256, lens[1]),
        (32, lens[2]),
    ] {
        let matrix = records(rows, len, rows as u64);

        // Plonky3's tree builder, batched leaves and all.
        let tree =
            MerkleTree::<u8, u8, _, 2, 32>::new::<u8, u8, _, _>(leaf, node, vec![matrix.clone()]);

        // The same tree, one record and one node at a time.
        let mut layer: Vec<[u8; 32]> = matrix.rows().map(|r| leaf.hash_iter(r)).collect();
        while layer.len() > 1 {
            layer = layer
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| node.compress(*pair))
                .collect();
        }

        let root: [u8; 32] = tree.root().into();
        assert_eq!(root, layer[0], "rows {rows} len {len}");
    }
}

/// Openings of a T8 tree verify, and a flipped record bit is rejected.
fn openings<H, C>(leaf: H, node: C, lens: [usize; 2])
where
    H: CryptographicHasher<u8, [u8; 32]> + Sync,
    C: PseudoCompressionFunction<[u8; 32], 2> + Sync,
{
    // Invariant: the opening is the whole record plus the usual sibling path.
    //
    // Fixture state: 128 records of 256 bytes, and a tree of 37 records that needs padding.
    let mmcs = MerkleTreeMmcs::<u8, u8, H, C, 2, 32>::new(leaf, node, 0);
    for (rows, len) in [(128, lens[0]), (37, lens[1])] {
        let matrix = records(rows, len, 7 + rows as u64);
        let dims = [matrix.dimensions()];
        let (commit, data) = mmcs.commit(vec![matrix]);

        for index in [0, 1, rows / 2, rows - 1] {
            let opening = mmcs.open_batch(index, &data);
            mmcs.verify_batch(&commit, &dims, index, (&opening).into())
                .expect("an honest opening verifies");

            // Mutation: flip one bit of the last block of the record.
            //
            //     record  [ z_1 .. z_7 | z_8 ]
            //                             ^ bit 0 of the last byte
            let mut values = opening.opened_values.clone();
            *values[0].last_mut().unwrap() ^= 1;
            let forged = BatchOpeningRef::new(&values, &opening.opening_proof);
            assert!(mmcs.verify_batch(&commit, &dims, index, forged).is_err());
        }
    }
}

#[test]
fn blake3_trees() {
    let node = CompressionFunctionFromHasher::<_, 2, 32>::new(Blake3);
    textbook_root(&T8Blake3, &node, T8_LENS);
    openings(T8Blake3, node, [T8_LENS[0], T8_LENS[1]]);
}

#[test]
fn blake2s_trees() {
    let node = CompressionFunctionFromHasher::<_, 2, 32>::new(Blake2s256);
    textbook_root(&T8Blake2s, &node, T8_LENS);
    openings(T8Blake2s, node, [T8_LENS[0], T8_LENS[1]]);
}

#[test]
fn sha256_trees() {
    textbook_root(&T8Sha256, &Sha256Compress, T8_LENS);
    openings(T8Sha256, Sha256Compress, [T8_LENS[0], T8_LENS[1]]);
}

#[test]
fn sha256_t253_trees() {
    textbook_root(&T253Sha256, &Sha256Compress, T253_LENS);
    openings(T253Sha256, Sha256Compress, [T253_LENS[0], T253_LENS[1]]);
}
