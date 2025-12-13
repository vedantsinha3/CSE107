# CSE 107 - Homework 4 Solutions

---

## Problem 1: Hough Transform

### Part (a): Equation for the Sinusoid at Pixel Location (0,0)

The Hough Transform converts points from image space to parameter space using the normal form of a line:

$$\rho = x \cos\theta + y \sin\theta$$

where:
- $\rho$ is the perpendicular distance from the origin to the line
- $\theta$ is the angle of the perpendicular from the origin

**For an edge point at pixel location (0,0):**

Substituting $x = 0$ and $y = 0$ into the equation:

$$\rho = 0 \cdot \cos\theta + 0 \cdot \sin\theta$$

$$\boxed{\rho = 0}$$

This means the sinusoid corresponding to the point (0,0) is simply the horizontal line $\rho = 0$ in parameter space. This line extends across all values of $\theta$ from $0$ to $\pi$ (or $-\pi/2$ to $\pi/2$ depending on the convention).

### Part (b): Other Points Corresponding to This Sinusoid

**No, there are no other points (at locations other than (0,0)) that correspond to this particular sinusoid.**

**Explanation:**

The sinusoid for any point $(x, y)$ in image space is given by:
$$\rho = x \cos\theta + y \sin\theta$$

For this equation to produce $\rho = 0$ for **all values of $\theta$** (which is what the sinusoid from (0,0) produces), we would need:

$$x \cos\theta + y \sin\theta = 0 \quad \forall \theta$$

This condition can only be satisfied when both $x = 0$ AND $y = 0$.

For any other point $(x, y) \neq (0, 0)$, the equation produces a true sinusoidal curve that varies with $\theta$, oscillating between positive and negative values of $\rho$. While these other sinusoids will *intersect* the line $\rho = 0$ at specific values of $\theta$, they do not *coincide* with it across all $\theta$.

**Key Insight:** The point (0,0) is unique because all lines passing through the origin have $\rho = 0$, so its "sinusoid" degenerates into a horizontal line in parameter space.

---

## Problem 2: Region Splitting and Merging (Quadtree Segmentation)

### Given:
- Image size: N = 8 (8×8 pixels)
- Predicate: $Q(R_i) = TRUE$ if all pixels in $R_i$ have the same intensity
- Image has two intensity values (white and gray)

### Interpreting the Image:

Based on the figure, the 8×8 image contains:
- **White pixels** (intensity 1): Background
- **Gray pixels** (intensity 0): An inverted-U shaped region (like a doorway)

The gray region occupies approximately:
- Columns 2-5 (indices 2,3,4,5)
- Rows 2-5 (indices 2,3,4,5)
- With a white rectangular opening at the bottom center

### Binary Representation of the Image:

```
Row\Col  0   1   2   3   4   5   6   7
  0      W   W   W   W   W   W   W   W
  1      W   W   W   W   W   W   W   W
  2      W   W   G   G   G   G   W   W
  3      W   W   G   W   W   G   W   W
  4      W   W   G   W   W   G   W   W
  5      W   W   G   G   G   G   W   W
  6      W   W   W   W   W   W   W   W
  7      W   W   W   W   W   W   W   W

W = White (intensity 1)
G = Gray (intensity 0)
```

### Quadtree Splitting Process:

**Level 0 (Root): R - Entire 8×8 image**
- Contains both white and gray pixels
- $Q(R) = FALSE$ → SPLIT

**Level 1: Split into four 4×4 quadrants**

| Quadrant | Position | Contents | Q(Ri) | Action |
|----------|----------|----------|-------|--------|
| R₁ | Top-left (rows 0-3, cols 0-3) | Mixed (W and G) | FALSE | SPLIT |
| R₂ | Top-right (rows 0-3, cols 4-7) | Mixed (W and G) | FALSE | SPLIT |
| R₃ | Bottom-left (rows 4-7, cols 0-3) | Mixed (W and G) | FALSE | SPLIT |
| R₄ | Bottom-right (rows 4-7, cols 4-7) | Mixed (W and G) | FALSE | SPLIT |

**Level 2: Split each 4×4 quadrant into 2×2 sub-quadrants**

**R₁ Sub-quadrants:**
| Region | Position | Contents | Q(Ri) | Action |
|--------|----------|----------|-------|--------|
| R₁₁ | rows 0-1, cols 0-1 | All White | TRUE | STOP |
| R₁₂ | rows 0-1, cols 2-3 | All White | TRUE | STOP |
| R₁₃ | rows 2-3, cols 0-1 | All White | TRUE | STOP |
| R₁₄ | rows 2-3, cols 2-3 | Mixed (G and W) | FALSE | SPLIT |

**R₂ Sub-quadrants:**
| Region | Position | Contents | Q(Ri) | Action |
|--------|----------|----------|-------|--------|
| R₂₁ | rows 0-1, cols 4-5 | All White | TRUE | STOP |
| R₂₂ | rows 0-1, cols 6-7 | All White | TRUE | STOP |
| R₂₃ | rows 2-3, cols 4-5 | Mixed (G and W) | FALSE | SPLIT |
| R₂₄ | rows 2-3, cols 6-7 | All White | TRUE | STOP |

**R₃ Sub-quadrants:**
| Region | Position | Contents | Q(Ri) | Action |
|--------|----------|----------|-------|--------|
| R₃₁ | rows 4-5, cols 0-1 | All White | TRUE | STOP |
| R₃₂ | rows 4-5, cols 2-3 | Mixed (G and W) | FALSE | SPLIT |
| R₃₃ | rows 6-7, cols 0-1 | All White | TRUE | STOP |
| R₃₄ | rows 6-7, cols 2-3 | All White | TRUE | STOP |

**R₄ Sub-quadrants:**
| Region | Position | Contents | Q(Ri) | Action |
|--------|----------|----------|-------|--------|
| R₄₁ | rows 4-5, cols 4-5 | Mixed (G and W) | FALSE | SPLIT |
| R₄₂ | rows 4-5, cols 6-7 | All White | TRUE | STOP |
| R₄₃ | rows 6-7, cols 4-5 | All White | TRUE | STOP |
| R₄₄ | rows 6-7, cols 6-7 | All White | TRUE | STOP |

**Level 3: Split remaining 2×2 regions into 1×1 pixels**

**R₁₄ (rows 2-3, cols 2-3):**
- R₁₄₁: (2,2) = G
- R₁₄₂: (2,3) = G
- R₁₄₃: (3,2) = G
- R₁₄₄: (3,3) = W

**R₂₃ (rows 2-3, cols 4-5):**
- R₂₃₁: (2,4) = G
- R₂₃₂: (2,5) = G
- R₂₃₃: (3,4) = W
- R₂₃₄: (3,5) = G

**R₃₂ (rows 4-5, cols 2-3):**
- R₃₂₁: (4,2) = G
- R₃₂₂: (4,3) = W
- R₃₂₃: (5,2) = G
- R₃₂₄: (5,3) = G

**R₄₁ (rows 4-5, cols 4-5):**
- R₄₁₁: (4,4) = W
- R₄₁₂: (4,5) = G
- R₄₁₃: (5,4) = G
- R₄₁₄: (5,5) = G

---

### Part (a): Segmented Image with Labeled Quadrants

```
┌───────────────┬───────────────┐
│               │               │
│     R₁₁      │     R₁₂      │     R₂₁      │     R₂₂      │
│               │               │               │               │
├───────┬───────┼───────┬───────┼───────┬───────┼───────────────┤
│       │R₁₄₁  │R₁₄₂  │R₂₃₁  │R₂₃₂  │               │
│ R₁₃  │───────┼───────┼───────┼───────│     R₂₄      │
│       │R₁₄₃  │R₁₄₄  │R₂₃₃  │R₂₃₄  │               │
├───────┼───────┼───────┼───────┼───────┼───────────────┤
│       │R₃₂₁  │R₃₂₂  │R₄₁₁  │R₄₁₂  │               │
│ R₃₁  │───────┼───────┼───────┼───────│     R₄₂      │
│       │R₃₂₃  │R₃₂₄  │R₄₁₃  │R₄₁₄  │               │
├───────┴───────┼───────────────┼───────────────┴───────────────┤
│               │               │               │               │
│     R₃₃      │     R₃₄      │     R₄₃      │     R₄₄      │
│               │               │               │               │
└───────────────┴───────────────┴───────────────┴───────────────┘
```

**Detailed Grid View:**

```
    Col: 0    1    2    3    4    5    6    7
       ┌─────────┬─────────┬─────────┬─────────┐
Row 0  │         │         │         │         │
       │   R₁₁   │   R₁₂   │   R₂₁   │   R₂₂   │
Row 1  │   (W)   │   (W)   │   (W)   │   (W)   │
       ├────┬────┼────┬────┼────┬────┼─────────┤
Row 2  │R₁₄₁│R₁₄₂│R₂₃₁│R₂₃₂│         │
       │ G  │ G  │ G  │ G  │   R₂₄   │
       │R₁₃ ├────┼────┼────┼────┤   (W)   │
Row 3  │(W) │R₁₄₃│R₁₄₄│R₂₃₃│R₂₃₄│         │
       │    │ G  │ W  │ W  │ G  │         │
       ├────┼────┼────┼────┼────┼─────────┤
Row 4  │    │R₃₂₁│R₃₂₂│R₄₁₁│R₄₁₂│         │
       │R₃₁ │ G  │ W  │ W  │ G  │   R₄₂   │
       │(W) ├────┼────┼────┼────┤   (W)   │
Row 5  │    │R₃₂₃│R₃₂₄│R₄₁₃│R₄₁₄│         │
       │    │ G  │ G  │ G  │ G  │         │
       ├────┴────┼─────────┼─────────┴─────────┤
Row 6  │         │         │         │         │
       │   R₃₃   │   R₃₄   │   R₄₃   │   R₄₄   │
Row 7  │   (W)   │   (W)   │   (W)   │   (W)   │
       └─────────┴─────────┴─────────┴─────────┘
```

---

### Part (b): Quadtree Structure

```
                                    R
                    ┌───────┬───────┼───────┬───────┐
                    │       │       │       │       │
                   R₁      R₂      R₃      R₄
              ┌──┬──┼──┐ ┌──┼──┬──┐ ┌──┬──┼──┐ ┌──┼──┬──┐
              │  │  │  │ │  │  │  │ │  │  │  │ │  │  │  │
            R₁₁ R₁₂ R₁₃ R₁₄ R₂₁ R₂₂ R₂₃ R₂₄ R₃₁ R₃₂ R₃₃ R₃₄ R₄₁ R₄₂ R₄₃ R₄₄
            (W) (W) (W)  │  (W) (W)  │  (W) (W)  │  (W) (W)  │  (W) (W) (W)
                    ┌──┬──┼──┬──┐    ┌──┬──┼──┬──┐    ┌──┬──┼──┬──┐    ┌──┬──┼──┬──┐
                    │  │  │  │      │  │  │  │      │  │  │  │      │  │  │  │
                  R₁₄₁R₁₄₂R₁₄₃R₁₄₄ R₂₃₁R₂₃₂R₂₃₃R₂₃₄ R₃₂₁R₃₂₂R₃₂₃R₃₂₄ R₄₁₁R₄₁₂R₄₁₃R₄₁₄
                   (G) (G) (G) (W)  (G) (G) (W) (G)  (G) (W) (G) (G)  (W) (G) (G) (G)
```

**Alternative Tree Representation:**

```
R (8×8) - Mixed
├── R₁ (4×4) - Mixed
│   ├── R₁₁ (2×2) - WHITE ✓
│   ├── R₁₂ (2×2) - WHITE ✓
│   ├── R₁₃ (2×2) - WHITE ✓
│   └── R₁₄ (2×2) - Mixed
│       ├── R₁₄₁ (1×1) - GRAY ✓
│       ├── R₁₄₂ (1×1) - GRAY ✓
│       ├── R₁₄₃ (1×1) - GRAY ✓
│       └── R₁₄₄ (1×1) - WHITE ✓
├── R₂ (4×4) - Mixed
│   ├── R₂₁ (2×2) - WHITE ✓
│   ├── R₂₂ (2×2) - WHITE ✓
│   ├── R₂₃ (2×2) - Mixed
│   │   ├── R₂₃₁ (1×1) - GRAY ✓
│   │   ├── R₂₃₂ (1×1) - GRAY ✓
│   │   ├── R₂₃₃ (1×1) - WHITE ✓
│   │   └── R₂₃₄ (1×1) - GRAY ✓
│   └── R₂₄ (2×2) - WHITE ✓
├── R₃ (4×4) - Mixed
│   ├── R₃₁ (2×2) - WHITE ✓
│   ├── R₃₂ (2×2) - Mixed
│   │   ├── R₃₂₁ (1×1) - GRAY ✓
│   │   ├── R₃₂₂ (1×1) - WHITE ✓
│   │   ├── R₃₂₃ (1×1) - GRAY ✓
│   │   └── R₃₂₄ (1×1) - GRAY ✓
│   ├── R₃₃ (2×2) - WHITE ✓
│   └── R₃₄ (2×2) - WHITE ✓
└── R₄ (4×4) - Mixed
    ├── R₄₁ (2×2) - Mixed
    │   ├── R₄₁₁ (1×1) - WHITE ✓
    │   ├── R₄₁₂ (1×1) - GRAY ✓
    │   ├── R₄₁₃ (1×1) - GRAY ✓
    │   └── R₄₁₄ (1×1) - GRAY ✓
    ├── R₄₂ (2×2) - WHITE ✓
    ├── R₄₃ (2×2) - WHITE ✓
    └── R₄₄ (2×2) - WHITE ✓
```

**Summary:**
- Total leaf nodes: 28
- White regions: 16 (12 at 2×2 level + 4 at 1×1 level)
- Gray regions: 12 (all at 1×1 level)

---

## Problem 3: Watershed Segmentation

### Given Cross-Section Data:

From the figure, the intensity values at each position are:

| Position (x) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|--------------|---|---|---|---|---|---|---|---|---|----|----|----|----|----|-----|
| Intensity    | 0 | 0 | 3 | 7 | 5 | 4 | 6 | 5 | 0 | 1  | 3  | 1  | 2  | 2  | 0   |

### Identifying Catchment Basins (Local Minima):

**Local minima locations:**
1. **Basin 1:** x = 1-2 (intensity 0) - Left edge minimum
2. **Basin 2:** x = 9 (intensity 0) - Central minimum  
3. **Basin 3:** x = 15 (intensity 0) - Right edge minimum

### Step-by-Step Watershed Algorithm:

The algorithm "floods" from the bottom up, filling from local minima. Dams are built where water from different basins would meet.

---

#### **Step 0: Initial State (Water Level = 0)**

```
Intensity
    7 ─          ┌─┐
    6 ─          │ │   ┌─┐
    5 ─        ┌─┘ │   │ └─┐
    4 ─        │   └───┘   │
    3 ─    ┌───┘           │   ┌─┐
    2 ─    │               │   │ └───┐
    1 ─    │               │ ┌─┘     │
    0 ─ ▓▓▓┘               └▓┘       └▓▓▓
        ───┴─┴─┴─┴─┴─┴─┴─┴─┴─┴──┴──┴──┴──┴──┴───→ x
        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15

▓ = Water at level 0 (fills minima at x=1,2,9,15)
```

Three basins begin filling:
- Basin 1: x = 1, 2
- Basin 2: x = 9
- Basin 3: x = 15

---

#### **Step 1: Water Level = 1**

```
Intensity
    7 ─          ┌─┐
    6 ─          │ │   ┌─┐
    5 ─        ┌─┘ │   │ └─┐
    4 ─        │   └───┘   │
    3 ─    ┌───┘           │   ┌─┐
    2 ─    │               │   │ └───┐
    1 ─ ░░░│               │ ░░┘     │░░░
    0 ─ ▓▓▓┘               └▓▓       └▓▓▓
        ───┴─┴─┴─┴─┴─┴─┴─┴─┴─┴──┴──┴──┴──┴──┴───→ x
        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15

░ = Water at level 1
▓ = Water at level 0
```

- Basin 1 expands slightly (still x = 1, 2)
- Basin 2 expands to x = 9, 10, 12 (since intensity at x=10 and x=12 is 1)
- Basin 3 expands to x = 14, 15

---

#### **Step 2: Water Level = 2**

```
Intensity
    7 ─          ┌─┐
    6 ─          │ │   ┌─┐
    5 ─        ┌─┘ │   │ └─┐
    4 ─        │   └───┘   │
    3 ─    ┌───┘           │   ┌─┐
    2 ─ ░░░│               │ ░░│ └░░░│░░░
    1 ─ ░░░│               │ ░░┘     │░░░
    0 ─ ▓▓▓┘               └▓▓       └▓▓▓
        ───┴─┴─┴─┴─┴─┴─┴─┴─┴─┴──┴──┴──┴──┴──┴───→ x
        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15

Water level 2: Basin 2 and Basin 3 would meet at x=13
BUILD DAM between x=12 and x=13!
```

**DAM 1 built between positions 12 and 13** to separate Basin 2 from Basin 3.

- Basin 2: x = 9, 10, 11, 12
- Basin 3: x = 13, 14, 15

---

#### **Step 3: Water Level = 3**

```
Intensity
    7 ─          ┌─┐
    6 ─          │ │   ┌─┐
    5 ─        ┌─┘ │   │ └─┐
    4 ─        │   └───┘   │
    3 ─ ░░░┌───┘           │ ░░┌▓┐░░░░░░░
    2 ─ ░░░│               │ ░░│▓│░░░│░░░
    1 ─ ░░░│               │ ░░┘▓    │░░░
    0 ─ ▓▓▓┘               └▓▓  ▓    └▓▓▓
        ───┴─┴─┴─┴─┴─┴─┴─┴─┴─┴──┴──┴──┴──┴──┴───→ x
        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
                                    ▓ = DAM
```

- Basin 1: x = 1, 2, 3
- Basin 2: x = 9, 10, 11, 12 (blocked by dam at right)
- Basin 3: x = 13, 14, 15 (blocked by dam at left)

---

#### **Step 4: Water Level = 4**

```
Intensity
    7 ─          ┌─┐
    6 ─          │ │   ┌─┐
    5 ─        ┌─┘ │   │ └─┐
    4 ─ ░░░░░░░│   └───┘   │░░░┌▓┐░░░░░░░
    3 ─ ░░░┌───┘           │░░░│▓│░░░░░░░
    2 ─ ░░░│               │░░░│▓│░░░│░░░
    1 ─ ░░░│               │░░░┘▓    │░░░
    0 ─ ▓▓▓┘               └▓▓▓ ▓    └▓▓▓
        ───┴─┴─┴─┴─┴─┴─┴─┴─┴─┴──┴──┴──┴──┴──┴───→ x
        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
```

- Basin 1: x = 1, 2, 3 (extends toward peak at x=4)
- Basin 2: x = 6, 8, 9, 10, 11, 12 (extends from x=6 and x=8)

Basin 1 and Basin 2 approaching each other...

---

#### **Step 5: Water Level = 5**

```
Intensity
    7 ─          ┌─┐
    6 ─          │ │   ┌─┐
    5 ─ ░░░░░░░┌─┘ │░░░│ └─┐░░░┌▓┐░░░░░░░
    4 ─ ░░░░░░░│   └───┘   │░░░│▓│░░░░░░░
    3 ─ ░░░┌───┘           │░░░│▓│░░░░░░░
    2 ─ ░░░│               │░░░│▓│░░░│░░░
    1 ─ ░░░│               │░░░┘▓    │░░░
    0 ─ ▓▓▓┘               └▓▓▓ ▓    └▓▓▓
        ───┴─┴─┴─┴─┴─┴─┴─┴─┴─┴──┴──┴──┴──┴──┴───→ x
        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15

Water from Basin 1 and Basin 2 would meet!
BUILD DAM between x=4 and x=5 (or at x=5)!
```

**DAM 2 built between positions 4 and 5** (at the ridge between the two basins)

---

#### **Step 6: Water Level = 6**

```
Intensity
    7 ─          ┌─┐
    6 ─ ░░░░░░░░▓│ │░░░┌─┐░░░░░┌▓┐░░░░░░░
    5 ─ ░░░░░░░┌▓┘ │░░░│ └─┐░░░│▓│░░░░░░░
    4 ─ ░░░░░░░│▓  └───┘   │░░░│▓│░░░░░░░
    3 ─ ░░░┌───┘▓          │░░░│▓│░░░░░░░
    2 ─ ░░░│    ▓          │░░░│▓│░░░│░░░
    1 ─ ░░░│    ▓          │░░░┘▓    │░░░
    0 ─ ▓▓▓┘    ▓          └▓▓▓ ▓    └▓▓▓
        ───┴─┴─┴─┴─┴─┴─┴─┴─┴─┴──┴──┴──┴──┴──┴───→ x
        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
                 ▓              ▓ = DAMS
```

Basin 2 would meet itself (from x=6 and x=8) at the ridge at x=7. This is within the same basin, so no new dam needed.

---

#### **Step 7: Water Level = 7 (Final)**

```
Intensity
    7 ─ ░░░░░░░░▓┌─┐░░░░░░░░░░░┌▓┐░░░░░░░
    6 ─ ░░░░░░░░▓│ │░░░┌─┐░░░░░│▓│░░░░░░░
    5 ─ ░░░░░░░┌▓┘ │░░░│ └─┐░░░│▓│░░░░░░░
    4 ─ ░░░░░░░│▓  └───┘   │░░░│▓│░░░░░░░
    3 ─ ░░░┌───┘▓          │░░░│▓│░░░░░░░
    2 ─ ░░░│    ▓          │░░░│▓│░░░│░░░
    1 ─ ░░░│    ▓          │░░░┘▓    │░░░
    0 ─ ▓▓▓┘    ▓          └▓▓▓ ▓    └▓▓▓
        ───┴─┴─┴─┴─┴─┴─┴─┴─┴─┴──┴──┴──┴──┴──┴───→ x
        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
```

---

### Final Segmentation Result:

**Three watershed regions (catchment basins) identified:**

| Segment | Positions | Color |
|---------|-----------|-------|
| **Basin 1** | x = 1, 2, 3, 4 | Left region |
| **Basin 2** | x = 5, 6, 7, 8, 9, 10, 11, 12 | Central region |
| **Basin 3** | x = 13, 14, 15 | Right region |

**Two dams (watershed lines) constructed:**

| Dam | Location | Separates |
|-----|----------|-----------|
| **Dam 1** | Between x = 4 and x = 5 | Basin 1 and Basin 2 |
| **Dam 2** | Between x = 12 and x = 13 | Basin 2 and Basin 3 |

---

### Summary Diagram:

```
    7 ┌────────────────────────────────────────────────────┐
      │          █                                         │
    6 │          █ █   █                                   │
      │        █ █ █   █ █                                 │
    5 │        █ █ █   █ █ █                               │
      │        █ █ █ █ █ █ █                               │
    4 │        █ █ █ █ █ █ █                  █            │
      │    █ █ █ █ █ █ █ █ █                  █ █ █        │
    3 │    █ █ █ █ █ █ █ █ █              █ █ █ █ █        │
      │    █ █ █ █ █ █ █ █ █          █ █ █ █ █ █ █        │
    2 │    █ █ █ █ █ █ █ █ █          █ █ █ █ █ █ █ █ █    │
      │    █ █ █ █ █ █ █ █ █        █ █ █ █ █ █ █ █ █ █    │
    1 │    █ █ █ █ █ █ █ █ █        █ █ █ █ █ █ █ █ █ █    │
      │ █ █ █ █ █ █ █ █ █ █ █    █ █ █ █ █ █ █ █ █ █ █ █ █ │
    0 │ █ █ █ █ D █ █ █ █ █ █ █ █ █ █ █ █ █ █ D █ █ █ █ █ │
      └────────────────────────────────────────────────────┘
        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        └──BASIN 1──┘└────────BASIN 2────────┘└─BASIN 3─┘
                    D = DAM                  D = DAM
```

The watershed segmentation successfully divides the 1D intensity profile into three distinct regions based on the local minima and the construction of dams at the points where water from different catchment basins would merge.

---

*End of Homework 4 Solutions*


