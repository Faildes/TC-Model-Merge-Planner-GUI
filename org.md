Hi, Jack's here

In this forum I'd like to share my method that is based on Lau method

Lau Method

Lau method goes something like this:

Base Edge Orientation (BEO/Cross-1)
I-F2L (Phase 1 Inverted-F2L, Phase 2 F2L Proper)
FLL (VHSL/VLS + COLL)
EPLL
While testing the method, I thought that somehow implementing Pseudo D layer aligning would have much faster figuration for solving

So, at the moment I solve 3x3 according to this step:

Pseudo 2x2x3 Building
FB-Line: Create line on either left side or right side by slotting its F and B slot
eg.) Left side >> Slot FL and BL, Right side >> Slot FR and BR
FBLine.png
D-Line: Create line on D layer with any opposite edges relates on D and rotate / place it as FD - BD Line
eg.) (Assume White as D) Both [White Blue and White Green] and [White Red and White Orange] can be placed as FD-BD Line
DLine.png
3rd Line: Slot the matching Corner-Edge-Corner 1x3x3 line to the D layer's opposite side of FB-Line and perform D2 afterwards to create pseudo 2x2x3 on the same side as FB-Line
Pseudo.png
OLL + F2L
E-EO: Insert any bad edges to the E slot that is placed on opposite side of FB-Line
After this, perform either d or d' and bring the pseudo 2x2x3 block to the bd
EEO.png
5EO: Orient the rest 5 edges using easy algorithms
5EO.png
eg.)
(Bad edges) 3U1D >> place the U bad edges to UL, UF and UR and perform: M' U' M or M' U M
1U1D >> place the U bad edge to UB and perform: M' U' M U' M' U' M or M' U M U M' U M
2U0D Opp (2U is placed on opposite sides like UF-UB) >> place the U bad edges to UF and UB and perform: F (R U R' U') F' (Think as T shape)
2U0D Adj (2U is placed on adjacent like UF-UR) >> place the U bad edges to UF and UR and perform: F (U R U' R') F' (Think as P shape)
4U0D >> most tricky one, but you can deduce this as 3U1D by doing: M' U2 M U2 after this, perform 3U1D
F2L + WV: Solve the rest slots and perform Winter Variation on last slot
Though using pairing would be difficult so I recommend separate recognition as E layer + D layer
Even if you don't know Winter Variation, doing the F2L results in CO + PLL state
PLL.png
LL: If you've done Winter Variation, this would be PLL
If you've done F2L, this would be either COLL + PLL or ZBLL
PLL2.png
ABF: Align both U and D face
Solve.png
Although this isn't better version of the Lau method, it could be great variant of it

Example solves:

(I'm not good at doing 3rd Line but I think you can do this by using the knowledge of Roux)

Scramble: F' U L R2 F2 D2 L2 D L2 U2 F' L' R2 D2 L R U

Solution:
B R U2 R' // FB-Line on right
l' U' l L2 U r2 // D-Line
L2 l' (U' L U L') l L U2 (L' U' L') D2 L2 D2 // 3rd Line
d // E-EO skip
U2 F U R U' R' F' // 3EO 2U0D Adj
U' r U2 r' D // FD + left slot
R U R' U2 R U' R' U L' U2 R U R' U2 L // right slot WV
M2 U' M U2 M' U' M2 // PLL
U2 D' // ABF

alg.cubing.net

Scramble: U2 L F' D R' U2 D' B F2 L2 D B2 D2 L2 B2 L2 U' L2 U' L2

Solution:
R U2 R // FB-Line on right
l' U l // D-Line
L' U' L U2 L' UD' L' U' L D' U2 L' U' L U' L2 D2 // 3rd Line
d // E-EO skip
U' M' U' M // 5EO 3U1D
U M' U2 M D R U2 R' D' // FD + left slot
U R U R' U2 R U' R' // right slot
U' (R U' L' U R' U' L) // ZBLL
D' // ABF

alg.cubing.net

Scramble: B2 U F2 U2 L2 D L2 U L2 R' B2 U' B' L D' R D' R' B'

Solution:
L F L2 // FB-Line on left
R' U R U' l2 // D-Line
U2 R' U' R U2 R' U R' U R D2 R2 D2 // 3rd Line
R' U R d' // E-EO
M' U M U M' U M // 5EO 1U1D
R U R' F2 R U' R' U2 R U2 R' U2 R U R' // FD + right slot
U' L' U' L U' L' U' L U L' U2 L U' L' U2 L // left slot WV
U' (F R U' R' U R U R2 F' R U R U' R') // PLL
D2 // ABF

alg.cubing.net
Thanks for reading!
