#ifdef __IPU__

#include "poplar/StackSizeDefs.hpp"

/**
 *  Codelets for candidate sampler ops
 *
 * These algorithms are simple implementations of sequential sampling from a
 * distribution on the range [0, N), discarding duplicates.
 * - We fill a tensor of 'samples' with unique samples from the distribution.
 * - We store the number of total samples we took into the numTries tensor, for
 *   future expectation calculations.
 * - We pass in a number of 32-bit integers representing one large bitmask,
 *   which allow us to lookup samples in constant time, like so:
 *
 *
 *                    integer 2 (32-63)                integer 1 (0-31)
 *   ...0000 00000000100000000000000101000000 00001000000000000000000001000000
 *
 *   S=6                                                                ^ = 0
 *   S=38                             ^ = 1, already sampled.
 *
 *   The bitmask is made up of ceil(N/32) 32-bit integers. For any sample S,
 *   we work out the integer from the bitmask we need (S // 32) via the right
 *   shift operator (S >> 5) and the offset into that integer (S % 32) via the
 *   31 lowest order bits (S & 31). The sample's mask is then 1 << (S & 31).
 *   By ANDing the bitmask integer with the sample's mask, we can see if the
 *   sample has already been used. If not, we update the bitmask integer with
 *   the OR operator:
 *
 *   S = 38, mask = 1 << (S & 31) =             1000000
 *   integer bitmask = 00000000100000000000000101000000
 *   result = 00000000000000000000000001000000 which is non-zero, been sampled.
 *
 * - The random numbers are generated with the urand64 hardware instruction,
 *   which gives floats in [-0.5, 0.5]. We offset them by 0.5 to get floats in
 *   [0, 1], multiply them by N (scale) and cast them to integers.
 *
 * - We use floats for the RNG so we don't hit representability problems when
 *   N is large (e.g. if N is >2048 and our sample is half type, we would never
 *   sample 2049 since it cannot be represented in half).
 *
 * - We don't simply store the bitmasks on the stack since there can be a lot of
 *   them depending on N.
 */

#define mOutAddr m0       // Base address of samples vertex
#define mOutSize m1       // Size of samples vertex
#define mBitmasksAddr m2  // Base address of bitmask integers
#define mNumTries m3      // Register for number of total samples taken
#define mNumTriesAddr m4  // Base address of num tries tensor
#define mRNGMask m5       // Use m5 to create the mask for the random numbers
#define mScratch m6       // Scratch register for temporary storage
#define mBitmask m7       // Save the integer bitmask for the random number
#define mRandOut0 m8      // Random number 1
#define mRandOut1 m9      // Random number 2
#define mBitmaskIdx m10   // Save the index of the integer bitmask we're using

#define aRandOut a0:1     // Aux random numbers 1 and 2
#define aRandOut0 a0      // Aux random number 1
#define aRandOut1 a1      // Aux random number 2
#define aShift a2         // Aux shift constant (0.5) to adjust RNG
#define aScale a3         // Aux scale (N) given to the codelet

// The vertex is given a set of inputs + outputs at $mvertex_base
// These are the offsets of each specific input/output we need
// The ordering must match that in the cpp codelet
#define VBASE_SCALE_OFFSET 0
#define VBASE_NUM_TRIES_OFFSET 1
#define VBASE_OUTPUT_BASE_OFFSET 2
#define VBASE_OUTPUT_SIZE_OFFSET 3
#define VBASE_BITMASKS_BASE_OFFSET 4

// Define the uniform unique codelet mangled symbol as something easier to read
#define uniformSeqSample __runCodelet_UniformUniqueSeqSample
.globl uniformSeqSample            // Allow access to the symbol from outside
.type uniformSeqSample, @function  // The symbol is a function
DEF_STACK_USAGE 0 uniformSeqSample // The symbol doesn't use any stack
.section .text.uniformSeqSample    // What follows is the body of the symbol
.align 4                           // Insert noops so that the body is 4 aligned

uniformSeqSample: // Entry point
  // Load the input/output tensors/addresses from memory
  ld32 $aScale, $mzero, $mvertex_base, VBASE_SCALE_OFFSET
  ld32 $mOutAddr, $mzero, $mvertex_base, VBASE_OUTPUT_BASE_OFFSET
  ld32 $mOutSize, $mzero, $mvertex_base, VBASE_OUTPUT_SIZE_OFFSET
  ld32 $mBitmasksAddr, $mzero, $mvertex_base, VBASE_BITMASKS_BASE_OFFSET

  // Set num tries to 0 and create a 0.5 constant for RNG
  setzi $mNumTries, 0
  ldconst $aShift, 0x3F000000 // 0.5

.LuniformSeqSampleLoop: // Main RNG loop
  // Create a random number = S
  urand64 $aRandOut                         // Create 2 32-bit random uintegers
  f32v2sufromui $aRandOut, $aRandOut        // uints to floats in [-0.5,0.5]
  f32v2add $aRandOut, $aShift:B, $aRandOut  // offset [-0.5, 0.5] -> [0, 1]
  f32v2mul $aRandOut, $aScale:B, $aRandOut  // scale [0, 1] -> [0, N)
  f32toui32 $aRandOut0, $aRandOut0          // floats to 32-bit uints
  atom $mRandOut0, $aRandOut0               // move one from aux to main
  add $mNumTries, $mNumTries, 1             // We sampled 1 more random number

  // Create 32-bit RNGMask for the random number (S % 32)
  // mask = (1 << (S & 31))
  setzi $mRNGMask, 1
  and $mScratch, $mRandOut0, 31
  shl $mRNGMask, $mRNGMask, $mScratch

  // Fetch the right integer bitmask from memory
  shr $mBitmaskIdx, $mRandOut0, 5                        // (S // 32)
  ld32 $mBitmask, $mzero, $mBitmasksAddr, $mBitmaskIdx   // bitmasks[S // 32]
  // AND with the RNG mask to see if we've sampled it
  and $mScratch, $mBitmask, $mRNGMask
  brnz $mScratch, .LuniformSeqSampleLoop

  // If not, update the bitmask and store it
  or $mBitmask, $mBitmask, $mRNGMask
  stm32 $mBitmask, $mBitmasksAddr, $mBitmaskIdx
  // Store S into the samples tensor and auto-increment our pointer into it
  st32step $mRandOut0, $mzero, $mOutAddr+=, 1
  // Check if we've filled the samples tensor
  sub $mOutSize, $mOutSize, 1
  brnz $mOutSize, .LuniformSeqSampleLoop

  // If we have, finish by storing num tries into the numTries tensor
  ld32 $mNumTriesAddr, $mzero, $mvertex_base, VBASE_NUM_TRIES_OFFSET
  st32 $mNumTries, $mzero, $mNumTriesAddr, 0
  exitz $mzero // terminate
.size uniformSeqSample, .-uniformSeqSample


// Define the log uniform unique codelet mangled symbol as easier to read
#define logUniformSeqSample __runCodelet_LogUniformUniqueSeqSample
.globl logUniformSeqSample            // Allow access to the symbol from outside
.type logUniformSeqSample, @function  // The symbol is a function
DEF_STACK_USAGE 0 logUniformSeqSample // The symbol doesn't use any stack
.section .text.logUniformSeqSample    // What follows is the body of the symbol
.align 4                              // Insert noops so that body is 4 aligned

logUniformSeqSample: // Entry point
  // Load the input/output tensors/addresses from memory
  ld32 $aScale, $mzero, $mvertex_base, VBASE_SCALE_OFFSET
  ld32 $mOutAddr, $mzero, $mvertex_base, VBASE_OUTPUT_BASE_OFFSET
  ld32 $mOutSize, $mzero, $mvertex_base, VBASE_OUTPUT_SIZE_OFFSET
  ld32 $mBitmasksAddr, $mzero, $mvertex_base, VBASE_BITMASKS_BASE_OFFSET

  // Set num tries to 0 and create a 0.5 constant for RNG
  setzi $mNumTries, 0
  ldconst $aShift, 0x3F000000 // 0.5

.LlogUniformSeqSampleLoop: // Main RNG loop
  // Create a random number = S
  urand64 $aRandOut                         // Create 2 32-bit random uintegers
  f32v2sufromui $aRandOut, $aRandOut        // uints to floats in [-0.5,0.5]
  f32v2add $aRandOut, $aShift:B, $aRandOut  // offset [-0.5, 0.5] -> [0, 1]
  f32v2mul $aRandOut, $aScale:B, $aRandOut  // scale [0, 1] -> [0, log(N)]
  f32exp $aRandOut0, $aRandOut0             // exp([0, log(N)]) -> [1, N]
  f32toui32 $aRandOut0, $aRandOut0          // floats to 32-bit uints
  atom $mRandOut0, $aRandOut0               // move one from aux to main
  sub $mRandOut0, $mRandOut0, 1             // [1, N] -> [0, N)

  add $mNumTries, $mNumTries, 1             // We sampled 1 more random number

  // Create 32-bit RNGMask for the random number (S % 32)
  // mask = (1 << (S & 31))
  setzi $mRNGMask, 1
  and $mScratch, $mRandOut0, 31
  shl $mRNGMask, $mRNGMask, $mScratch

  // Fetch the right integer bitmask from memory
  shr $mBitmaskIdx, $mRandOut0, 5                        // (S // 32)
  ld32 $mBitmask, $mzero, $mBitmasksAddr, $mBitmaskIdx   // bitmasks[S // 32]
  // AND with the RNG mask to see if we've sampled it
  and $mScratch, $mBitmask, $mRNGMask
  brnz $mScratch, .LlogUniformSeqSampleLoop

  // If not, update the bitmask and store it
  or $mBitmask, $mBitmask, $mRNGMask
  stm32 $mBitmask, $mBitmasksAddr, $mBitmaskIdx
  // Store S into the samples tensor and auto-increment our pointer into it
  st32step $mRandOut0, $mzero, $mOutAddr+=, 1
  // Check if we've filled the samples tensor
  sub $mOutSize, $mOutSize, 1
  brnz $mOutSize, .LlogUniformSeqSampleLoop

  // If we have, finish by storing num tries into the numTries tensor
  ld32 $mNumTriesAddr, $mzero, $mvertex_base, VBASE_NUM_TRIES_OFFSET
  st32 $mNumTries, $mzero, $mNumTriesAddr, 0
  exitz $mzero // terminate
.size logUniformSeqSample, .-logUniformSeqSample

#endif  // __IPU__
