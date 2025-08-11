//! Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323
//!
//! For the diffusion matrix, 1 + Diag(V), we perform a search to find an optimized
//! vector V composed of elements with efficient multiplication algorithms in AVX2/AVX512/NEON.
//!
//! This leads to using small values (e.g. 1, 2, 3, 4) where multiplication is implemented using addition
//! and inverse powers of 2 where it is possible to avoid monty reductions.
//! Additionally, for technical reasons, having the first entry be -2 is useful.
//!
//! Optimized Diagonal for KoalaBear16:
//! [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24]
//! Optimized Diagonal for KoalaBear24:
//! [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/32, 1/64, 1/2^24, -1/2^8, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^9, -1/2^24]
//! See poseidon2\src\diffusion.rs for information on how to double check these matrices in Sage.

use p3_field::{Algebra, Field, PrimeCharacteristicRing, PrimeField32};
use p3_monty_31::{
    GenericPoseidon2LinearLayersMonty31, InternalLayerBaseParameters, InternalLayerParameters,
    MontyField31, Poseidon2ExternalLayerMonty31, Poseidon2InternalLayerMonty31,
};
use p3_poseidon2::Poseidon2;

use crate::{KoalaBear, KoalaBearParameters};

pub type Poseidon2InternalLayerKoalaBear<const WIDTH: usize> =
    Poseidon2InternalLayerMonty31<KoalaBearParameters, WIDTH, KoalaBearInternalLayerParameters>;

pub type Poseidon2ExternalLayerKoalaBear<const WIDTH: usize> =
    Poseidon2ExternalLayerMonty31<KoalaBearParameters, WIDTH>;

/// Degree of the chosen permutation polynomial for KoalaBear, used as the Poseidon2 S-Box.
///
/// As p - 1 = 127 * 2^{24} we have a lot of choice in degree D satisfying gcd(p - 1, D) = 1.
/// Experimentation suggests that the optimal choice is the smallest available one, namely 3.
const KOALABEAR_S_BOX_DEGREE: u64 = 3;

/// An implementation of the Poseidon2 hash function specialised to run on the current architecture.
///
/// It acts on arrays of the form either `[KoalaBear::Packing; WIDTH]` or `[KoalaBear; WIDTH]`. For speed purposes,
/// wherever possible, input arrays should of the form `[KoalaBear::Packing; WIDTH]`.
pub type Poseidon2KoalaBear<const WIDTH: usize> = Poseidon2<
    KoalaBear,
    Poseidon2ExternalLayerKoalaBear<WIDTH>,
    Poseidon2InternalLayerKoalaBear<WIDTH>,
    WIDTH,
    KOALABEAR_S_BOX_DEGREE,
>;

/// An implementation of the matrix multiplications in the internal and external layers of Poseidon2.
///
/// This can act on `[A; WIDTH]` for any ring implementing `Algebra<BabyBear>`.
/// If you have either `[KoalaBear::Packing; WIDTH]` or `[KoalaBear; WIDTH]` it will be much faster
/// to use `Poseidon2KoalaBear<WIDTH>` instead of building a Poseidon2 permutation using this.
pub type GenericPoseidon2LinearLayersKoalaBear =
    GenericPoseidon2LinearLayersMonty31<KoalaBearParameters, KoalaBearInternalLayerParameters>;

// In order to use KoalaBear::new_array we need to convert our vector to a vector of u32's.
// To do this we make use of the fact that KoalaBear::ORDER_U32 - 1 = 127 * 2^24 so for 0 <= n <= 24:
// -1/2^n = (KoalaBear::ORDER_U32 - 1) >> n
// 1/2^n = -(-1/2^n) = KoalaBear::ORDER_U32 - ((KoalaBear::ORDER_U32 - 1) >> n)

/// The vector [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24]
/// saved as an array of KoalaBear elements.
const INTERNAL_DIAG_MONTY_16: [KoalaBear; 16] = KoalaBear::new_array([
    KoalaBear::ORDER_U32 - 2,
    1,
    2,
    (KoalaBear::ORDER_U32 + 1) >> 1,
    3,
    4,
    (KoalaBear::ORDER_U32 - 1) >> 1,
    KoalaBear::ORDER_U32 - 3,
    KoalaBear::ORDER_U32 - 4,
    KoalaBear::ORDER_U32 - ((KoalaBear::ORDER_U32 - 1) >> 8),
    KoalaBear::ORDER_U32 - ((KoalaBear::ORDER_U32 - 1) >> 3),
    KoalaBear::ORDER_U32 - 127,
    (KoalaBear::ORDER_U32 - 1) >> 8,
    (KoalaBear::ORDER_U32 - 1) >> 3,
    (KoalaBear::ORDER_U32 - 1) >> 4,
    127,
]);

/// The vector [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/32, 1/64, 1/2^24, -1/2^8, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^9, -1/2^24]
/// saved as an array of KoalaBear elements.
const INTERNAL_DIAG_MONTY_24: [KoalaBear; 24] = KoalaBear::new_array([
    KoalaBear::ORDER_U32 - 2,
    1,
    2,
    (KoalaBear::ORDER_U32 + 1) >> 1,
    3,
    4,
    (KoalaBear::ORDER_U32 - 1) >> 1,
    KoalaBear::ORDER_U32 - 3,
    KoalaBear::ORDER_U32 - 4,
    KoalaBear::ORDER_U32 - ((KoalaBear::ORDER_U32 - 1) >> 8),
    KoalaBear::ORDER_U32 - ((KoalaBear::ORDER_U32 - 1) >> 2),
    KoalaBear::ORDER_U32 - ((KoalaBear::ORDER_U32 - 1) >> 3),
    KoalaBear::ORDER_U32 - ((KoalaBear::ORDER_U32 - 1) >> 4),
    KoalaBear::ORDER_U32 - ((KoalaBear::ORDER_U32 - 1) >> 5),
    KoalaBear::ORDER_U32 - ((KoalaBear::ORDER_U32 - 1) >> 6),
    KoalaBear::ORDER_U32 - 127,
    (KoalaBear::ORDER_U32 - 1) >> 8,
    (KoalaBear::ORDER_U32 - 1) >> 3,
    (KoalaBear::ORDER_U32 - 1) >> 4,
    (KoalaBear::ORDER_U32 - 1) >> 5,
    (KoalaBear::ORDER_U32 - 1) >> 6,
    (KoalaBear::ORDER_U32 - 1) >> 7,
    (KoalaBear::ORDER_U32 - 1) >> 9,
    127,
]);

/// Contains data needed to define the internal layers of the Poseidon2 permutation.
#[derive(Debug, Clone, Default)]
pub struct KoalaBearInternalLayerParameters;

impl InternalLayerBaseParameters<KoalaBearParameters, 16> for KoalaBearInternalLayerParameters {
    type ArrayLike = [MontyField31<KoalaBearParameters>; 15];

    const INTERNAL_DIAG_MONTY: [MontyField31<KoalaBearParameters>; 16] = INTERNAL_DIAG_MONTY_16;

    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul(
        state: &mut [MontyField31<KoalaBearParameters>; 16],
        sum: MontyField31<KoalaBearParameters>,
    ) {
        // The diagonal matrix is defined by the vector:
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24]
        state[1] += sum;
        state[2] = state[2].double() + sum;
        state[3] = state[3].halve() + sum;
        state[4] = sum + state[4].double() + state[4];
        state[5] = sum + state[5].double().double();
        state[6] = sum - state[6].halve();
        state[7] = sum - (state[7].double() + state[7]);
        state[8] = sum - state[8].double().double();
        state[9] = state[9].div_2exp_u64(8);
        state[9] += sum;
        state[10] = state[10].div_2exp_u64(3);
        state[10] += sum;
        state[11] = state[11].div_2exp_u64(24);
        state[11] += sum;
        state[12] = state[12].div_2exp_u64(8);
        state[12] = sum - state[12];
        state[13] = state[13].div_2exp_u64(3);
        state[13] = sum - state[13];
        state[14] = state[14].div_2exp_u64(4);
        state[14] = sum - state[14];
        state[15] = state[15].div_2exp_u64(24);
        state[15] = sum - state[15];
    }

    fn generic_internal_linear_layer<A: Algebra<KoalaBear>>(state: &mut [A; 16]) {
        let part_sum: A = state[1..].iter().cloned().sum();
        let full_sum = part_sum.clone() + state[0].clone();

        // The first three diagonal elements are -2, 1, 2 so we do something custom.
        state[0] = part_sum - state[0].clone();
        state[1] = full_sum.clone() + state[1].clone();
        state[2] = full_sum.clone() + state[2].double();

        // For the remaining elements we use multiplication.
        // This could probably be improved slightly by making use of the
        // mul_2exp_u64 and div_2exp_u64 but this would involve porting div_2exp_u64 to PrimeCharacteristicRing.
        state
            .iter_mut()
            .zip(INTERNAL_DIAG_MONTY_16)
            .skip(3)
            .for_each(|(val, diag_elem)| {
                *val = full_sum.clone() + val.clone() * diag_elem;
            });
    }
}

impl InternalLayerBaseParameters<KoalaBearParameters, 24> for KoalaBearInternalLayerParameters {
    type ArrayLike = [MontyField31<KoalaBearParameters>; 23];

    const INTERNAL_DIAG_MONTY: [MontyField31<KoalaBearParameters>; 24] = INTERNAL_DIAG_MONTY_24;

    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul(
        state: &mut [MontyField31<KoalaBearParameters>; 24],
        sum: MontyField31<KoalaBearParameters>,
    ) {
        // The diagonal matrix is defined by the vector:
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/32, 1/64, 1/2^24, -1/2^8, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^9, -1/2^24]
        state[1] += sum;
        state[2] = state[2].double() + sum;
        state[3] = state[3].halve() + sum;
        state[4] = sum + state[4].double() + state[4];
        state[5] = sum + state[5].double().double();
        state[6] = sum - state[6].halve();
        state[7] = sum - (state[7].double() + state[7]);
        state[8] = sum - state[8].double().double();
        state[9] = state[9].div_2exp_u64(8);
        state[9] += sum;
        state[10] = state[10].div_2exp_u64(2);
        state[10] += sum;
        state[11] = state[11].div_2exp_u64(3);
        state[11] += sum;
        state[12] = state[12].div_2exp_u64(4);
        state[12] += sum;
        state[13] = state[13].div_2exp_u64(5);
        state[13] += sum;
        state[14] = state[14].div_2exp_u64(6);
        state[14] += sum;
        state[15] = state[15].div_2exp_u64(24);
        state[15] += sum;
        state[16] = state[16].div_2exp_u64(8);
        state[16] = sum - state[16];
        state[17] = state[17].div_2exp_u64(3);
        state[17] = sum - state[17];
        state[18] = state[18].div_2exp_u64(4);
        state[18] = sum - state[18];
        state[19] = state[19].div_2exp_u64(5);
        state[19] = sum - state[19];
        state[20] = state[20].div_2exp_u64(6);
        state[20] = sum - state[20];
        state[21] = state[21].div_2exp_u64(7);
        state[21] = sum - state[21];
        state[22] = state[22].div_2exp_u64(9);
        state[22] = sum - state[22];
        state[23] = state[23].div_2exp_u64(24);
        state[23] = sum - state[23];
    }

    fn generic_internal_linear_layer<A: Algebra<KoalaBear>>(state: &mut [A; 24]) {
        let part_sum: A = state[1..].iter().cloned().sum();
        let full_sum = part_sum.clone() + state[0].clone();

        // The first three diagonal elements are -2, 1, 2 so we do something custom.
        state[0] = part_sum - state[0].clone();
        state[1] = full_sum.clone() + state[1].clone();
        state[2] = full_sum.clone() + state[2].double();

        // For the remaining elements we use multiplication.
        // This could probably be improved slightly by making use of the
        // mul_2exp_u64 and div_2exp_u64 but this would involve porting div_2exp_u64 to PrimeCharacteristicRing.
        state
            .iter_mut()
            .zip(INTERNAL_DIAG_MONTY_24)
            .skip(3)
            .for_each(|(val, diag_elem)| {
                *val = full_sum.clone() + val.clone() * diag_elem;
            });
    }
}

impl InternalLayerParameters<KoalaBearParameters, 16> for KoalaBearInternalLayerParameters {}
impl InternalLayerParameters<KoalaBearParameters, 24> for KoalaBearInternalLayerParameters {}

#[cfg(test)]
mod tests {
    use p3_poseidon2::ExternalLayerConstants;
    use p3_symmetric::Permutation;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    type F = KoalaBear;

    // We need to make some round constants. We use Xoroshiro128Plus for this as we can easily match this PRNG in sage.
    // See: https://github.com/0xPolygonZero/hash-constants for the sage code used to create all these tests.

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(16)
    /// vector([KB.random_element() for t in range(16)]).
    #[test]
    fn test_poseidon2_width_16_random() {
        let mut input: [F; 16] = KoalaBear::new_array([
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]);

        let expected: [F; 16] = KoalaBear::new_array([
            652590279, 1200629963, 1013089423, 1840372851, 19101828, 561050015, 1714865585,
            994637181, 498949829, 729884572, 1957973925, 263012103, 535029297, 2121808603,
            964663675, 1473622080,
        ]);

        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Poseidon2KoalaBear::new_from_rng_128(&mut rng);

        perm.permute_mut(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(24)
    /// vector([KB.random_element() for t in range(24)]).
    #[test]
    fn test_poseidon2_width_24_random() {
        let mut input: [F; 24] = KoalaBear::new_array([
            886409618, 1327899896, 1902407911, 591953491, 648428576, 1844789031, 1198336108,
            355597330, 1799586834, 59617783, 790334801, 1968791836, 559272107, 31054313,
            1042221543, 474748436, 135686258, 263665994, 1962340735, 1741539604, 2026927696,
            449439011, 1131357108, 50869465,
        ]);

        let expected: [F; 24] = KoalaBear::new_array([
            3825456, 486989921, 613714063, 282152282, 1027154688, 1171655681, 879344953,
            1090688809, 1960721991, 1604199242, 1329947150, 1535171244, 781646521, 1156559780,
            1875690339, 368140677, 457503063, 304208551, 1919757655, 835116474, 1293372648,
            1254825008, 810923913, 1773631109,
        ]);

        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Poseidon2KoalaBear::new_from_rng_128(&mut rng);

        perm.permute_mut(&mut input);
        assert_eq!(input, expected);
    }

    /// Test the generic internal layer against the optimized internal layer
    /// for a random input of width 16.
    #[test]
    fn test_generic_internal_linear_layer_16() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let mut input1: [F; 16] = rng.random();
        let mut input2 = input1;

        let part_sum: F = input1[1..].iter().copied().sum();
        let full_sum = part_sum + input1[0];

        input1[0] = part_sum - input1[0];

        KoalaBearInternalLayerParameters::internal_layer_mat_mul(&mut input1, full_sum);
        KoalaBearInternalLayerParameters::generic_internal_linear_layer(&mut input2);

        assert_eq!(input1, input2);
    }

    /// Test the generic internal layer against the optimized internal layer
    /// for a random input of width 16.
    #[test]
    fn test_generic_internal_linear_layer_24() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let mut input1: [F; 24] = rng.random();
        let mut input2 = input1;

        let part_sum: F = input1[1..].iter().copied().sum();
        let full_sum = part_sum + input1[0];

        input1[0] = part_sum - input1[0];

        KoalaBearInternalLayerParameters::internal_layer_mat_mul(&mut input1, full_sum);
        KoalaBearInternalLayerParameters::generic_internal_linear_layer(&mut input2);

        assert_eq!(input1, input2);
    }

    // --- 1. Use the canonical constants from the Python spec ---
    // Source: https://github.com/leanEthereum/leanSpec/issues/4#issuecomment-2287903820
    const RAW_CONSTANTS: [u64; 448] = [
        2128964168, 288780357, 316938561, 2126233899, 426817493, 1714118888, 1045008582,
        1738510837, 889721787, 8866516, 681576474, 419059826, 1596305521, 1583176088, 1584387047,
        1529751136, 1863858111, 1072044075, 517831365, 1464274176, 1138001621, 428001039,
        245709561, 1641420379, 1365482496, 770454828, 693167409, 757905735, 136670447, 436275702,
        525466355, 1559174242, 1030087950, 869864998, 322787870, 267688717, 948964561, 740478015,
        679816114, 113662466, 2066544572, 1744924186, 367094720, 1380455578, 1842483872, 416711434,
        1342291586, 1692058446, 1493348999, 1113949088, 210900530, 1071655077, 610242121,
        1136339326, 2020858841, 1019840479, 678147278, 1678413261, 1361743414, 61132629,
        1209546658, 64412292, 1936878279, 1980661727, 1423960925, 2101391318, 1915532054,
        275400051, 1168624859, 1141248885, 356546469, 1165250474, 1320543726, 932505663,
        1204226364, 1452576828, 1774936729, 926808140, 1184948056, 1186493834, 843181003,
        185193011, 452207447, 510054082, 1139268644, 630873441, 669538875, 462500858, 876500520,
        1214043330, 383937013, 375087302, 636912601, 307200505, 390279673, 1999916485, 1518476730,
        1606686591, 1410677749, 1581191572, 1004269969, 143426723, 1747283099, 1016118214,
        1749423722, 66331533, 1177761275, 1581069649, 1851371119, 852520128, 1499632627,
        1820847538, 150757557, 884787840, 619710451, 1651711087, 505263814, 212076987, 1482432120,
        1458130652, 382871348, 417404007, 2066495280, 1996518884, 902934924, 582892981, 1337064375,
        1199354861, 2102596038, 1533193853, 1436311464, 2012303432, 839997195, 1225781098,
        2011967775, 575084315, 1309329169, 786393545, 995788880, 1702925345, 1444525226, 908073383,
        1811535085, 1531002367, 1635653662, 1585100155, 867006515, 879151050, 1686691828,
        1911580916, 91130143, 82963660, 1714575317, 1730032057, 1483839612, 671879326, 706901857,
        889857513, 1536274884, 2047292742, 25322096, 1403418400, 248819828, 885984334, 1853169288,
        700276569, 1240216287, 1989362987, 1022402136, 1805705919, 2058959567, 1021679583,
        1399733570, 343572621, 1580395350, 1512059683, 1352030054, 1833220037, 1721262954,
        1471696799, 1431003577, 1839246120, 361084588, 1728422580, 354972406, 256117245, 598334816,
        1865095380, 1705811924, 789511146, 1495164925, 561815963, 1184665802, 64360181, 1319601534,
        130574927, 680449121, 803543842, 2116630036, 743172997, 1527479569, 504881142, 144435937,
        173723418, 801324431, 1614949830, 1847445817, 1666404793, 1431449536, 1052331767,
        707044956, 773037174, 1362694468, 2026637122, 1469397241, 1155439278, 1009720878,
        425150398, 613823388, 1695231545, 1384748645, 1823692120, 256252956, 1895215728,
        1068147567, 1659057290, 1730242507, 961316875, 709278338, 1677702986, 486045142,
        1406216050, 57296210, 1004379947, 49753124, 45482092, 125821272, 530411172, 546327919,
        1566913786, 107841908, 1637413364, 640686772, 1106408642, 15384924, 682969927, 590967709,
        1220945948, 1322857980, 1066502138, 1243164838, 987027254, 255793289, 1666857103,
        677560645, 662622696, 1303526573, 521867765, 524139051, 472312654, 260003142, 1825580208,
        1740929282, 2033944832, 243935292, 1167112170, 1867938347, 1573483264, 354712518,
        1347846091, 322895748, 1417528047, 887831995, 306193175, 1724296777, 390281398, 606408712,
        458311975, 103651542, 2062748604, 649008616, 1893271459, 1576819884, 1931421676,
        1403682111, 1672154822, 559961076, 410610489, 420834045, 1592420723, 1728366249, 231604267,
        856779200, 1900900728, 1037762479, 2118535511, 550132202, 1738023113, 1122967969,
        2039390345, 346509219, 201772824, 1783401810, 1645178241, 572559386, 1383578512, 587987294,
        181961850, 1586948278, 2008286574, 1889865004, 1594813785, 910607583, 283875975, 569300663,
        1397415222, 1849586721, 723878158, 495939707, 1160874522, 1736413170, 39373280, 1288710656,
        774176533, 1665823069, 1254104665, 1611993569, 652853274, 1276533870, 1473057088,
        986076219, 1736955975, 58588153, 1842991225, 1294250625, 711934077, 20045710, 1267366038,
        594544728, 754312500, 313195583, 1414958339, 438634293, 1395746925, 1290235281, 2040273548,
        729451209, 1622074994, 1962361372, 1010963565, 651389381, 1256540690, 2129270481,
        1558440680, 1777502612, 640386626, 1628261572, 1578824220, 444933840, 829100667, 896990813,
        47802528, 1268780881, 1086249363, 931117319, 2019107182, 422697425, 1404080974, 1905348599,
        1319874156, 1905673870, 374029506, 1489725120, 1276408583, 1799027917, 1110856075,
        1255691781, 689144545, 512341711, 1578550184, 778524961, 607127892, 98915779, 2022181412,
        1983525157, 1330885184, 414710339, 733907571, 479859442, 1064293389, 236801732, 325174861,
        162067568, 64109120, 278581904, 683867016, 996448498, 1960361559, 1782740946, 415413204,
        1649591052, 130819424, 547348827, 1386569644, 1307680439, 38932758, 1581338609, 1020895732,
        5942549, 665140992, 1924917707, 1910029693, 1100265370, 1223195250, 859919676, 1674792874,
        321520099, 942924505, 1232236036, 88692728, 2071051492, 1945027965, 1433294131, 531185630,
        879398056, 291692510, 1546702888, 155861652, 810736858, 932742296, 1374710679, 1703184249,
        1973006548, 1131403964, 1724233597, 1086876318, 669451611, 1829624280, 2119538869,
        441255155, 1580936135, 1396398895, 1043570981, 1716351438, 942566442, 616885102, 334644983,
        132306927,
    ];

    /// Generates test vectors for the Python spec using sequential round constants.
    /// This test builds a Poseidon2 instance with constants [0, 1, 2, ...] to
    /// match the simple generation in the Python spec, then runs it on a known
    /// input. The resulting output can be copied into the Python tests to ensure
    /// both implementations are perfectly aligned.
    #[test]
    fn test_spec_vectors_width_16() {
        const WIDTH: usize = 16;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 20;
        let half_rounds_f = ROUNDS_F / 2;

        let num_constants_needed = (ROUNDS_F * WIDTH) + ROUNDS_P;
        let mut const_iter = RAW_CONSTANTS[..num_constants_needed]
            .iter()
            .map(|&c| F::from_u64(c));

        // --- 2. Split constants into the structure Rust's Poseidon2::new expects ---
        let initial_constants: Vec<[F; WIDTH]> = (0..half_rounds_f)
            .map(|_| {
                let mut round_consts = [F::ZERO; WIDTH];
                for i in 0..WIDTH {
                    round_consts[i] = const_iter.next().unwrap();
                }
                round_consts
            })
            .collect();

        let internal_constants: Vec<F> = const_iter.by_ref().take(ROUNDS_P).collect();

        let terminal_constants: Vec<[F; WIDTH]> = (0..half_rounds_f)
            .map(|_| {
                let mut round_consts = [F::ZERO; WIDTH];
                for i in 0..WIDTH {
                    round_consts[i] = const_iter.next().unwrap();
                }
                round_consts
            })
            .collect();

        // --- 3. Create the permutation with the specified constants ---
        let external_constants = ExternalLayerConstants::new(initial_constants, terminal_constants);
        let perm = Poseidon2KoalaBear::<WIDTH>::new(external_constants, internal_constants);

        // --- 4. Define the input vector (same as the Python test) ---
        let mut input: [F; 16] = KoalaBear::new_array([
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]);

        // --- 5. Run the permutation and print the output ---
        perm.permute_mut(&mut input);

        println!("\n--- Rust Output for Python Spec (Width 16) ---");
        println!("EXPECTED_16 = [");
        for (i, val) in input.iter().enumerate() {
            println!("    Fp(value={}),", val.as_canonical_u32());
        }
        println!("]");
    }

    /// Generates test vectors for the Python spec with WIDTH=24.
    #[test]
    fn test_spec_vectors_width_24() {
        const WIDTH: usize = 24;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 23;
        let half_rounds_f = ROUNDS_F / 2;

        // For width 24, we need 8*24 + 23 = 215 constants.
        // We take them starting after the 148 used for width 16.
        let num_constants_needed = (ROUNDS_F * WIDTH) + ROUNDS_P;
        let start_index = 148;
        let mut const_iter = RAW_CONSTANTS[start_index..start_index + num_constants_needed]
            .iter()
            .map(|&c| F::from_u64(c));

        // --- 2. Split constants into the required structure ---
        let initial_constants: Vec<[F; WIDTH]> = (0..half_rounds_f)
            .map(|_| {
                let mut round_consts = [F::ZERO; WIDTH];
                for i in 0..WIDTH {
                    round_consts[i] = const_iter.next().unwrap();
                }
                round_consts
            })
            .collect();

        let internal_constants: Vec<F> = const_iter.by_ref().take(ROUNDS_P).collect();

        let terminal_constants: Vec<[F; WIDTH]> = (0..half_rounds_f)
            .map(|_| {
                let mut round_consts = [F::ZERO; WIDTH];
                for i in 0..WIDTH {
                    round_consts[i] = const_iter.next().unwrap();
                }
                round_consts
            })
            .collect();

        // --- 3. Create the permutation ---
        let external_constants = ExternalLayerConstants::new(initial_constants, terminal_constants);
        let perm = Poseidon2KoalaBear::<WIDTH>::new(external_constants, internal_constants);

        // --- 4. Define the input vector ---
        let mut input: [F; 24] = KoalaBear::new_array([
            886409618, 1327899896, 1902407911, 591953491, 648428576, 1844789031, 1198336108,
            355597330, 1799586834, 59617783, 790334801, 1968791836, 559272107, 31054313,
            1042221543, 474748436, 135686258, 263665994, 1962340735, 1741539604, 2026927696,
            449439011, 1131357108, 50869465,
        ]);

        // --- 5. Run the permutation and print the output ---
        perm.permute_mut(&mut input);

        println!("\n--- Rust Output for Python Spec (Width 24) ---");
        println!("EXPECTED_24 = [");
        for (i, val) in input.iter().enumerate() {
            println!("    Fp(value={}),", val.as_canonical_u32());
        }
        println!("]");
    }
}
