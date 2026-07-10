# 📖 Data Dictionary

Source: `data/raw/cars24-car-price-cleaned-new.csv.gz` (gzip-compressed; pandas
reads it directly) — **19,820 cleaned Cars24
used-car listings**, 17 columns, covering **41 makes** and **3,233 models**.

## Columns

| Column | Type | Role | Meaning |
| :----- | :--- | :--- | :------ |
| `selling_price` | float | **target** | Resale price in **Lakhs of ₹** (e.g. `4.75` = ₹4.75 L). |
| `make` | string | feature (target-encoded) | Brand, e.g. `MARUTI`. 41 distinct. |
| `model` | string | feature (target-encoded) | Variant, e.g. `SWIFT VXI`. 3,233 distinct. |
| `km_driven` | int | feature (numeric) | Odometer reading. |
| `mileage` | float | feature (numeric) | Fuel efficiency (km/l). |
| `engine` | int | feature (numeric) | Engine displacement (cc). |
| `max_power` | float | feature (numeric) | Peak power (bhp). |
| `age` | int | feature (numeric) | Car age in years. |
| `Individual` | 0/1 | feature (flag) | Seller = Individual. |
| `Trustmark Dealer` | 0/1 | feature (flag) | Seller = Trustmark Dealer. |
| `Diesel`, `Electric`, `LPG`, `Petrol` | 0/1 | feature (flags) | Fuel type. |
| `Manual` | 0/1 | feature (flag) | Transmission = Manual. |
| `5` → `Seats_5` | 0/1 | feature (flag) | Exactly 5 seats. |
| `>5` → `Seats_Above_5` | 0/1 | feature (flag) | More than 5 seats. |

## The dropped-baseline convention

The categorical groups are one-hot encoded with **one level omitted** (the
baseline), so "all flags 0" is itself a meaningful category:

| Group | Flags present | Dropped baseline (all-zero) |
| :---- | :------------ | :-------------------------- |
| Seller | `Individual`, `Trustmark Dealer` | **Dealer** (~60 % of rows) |
| Fuel | `Petrol`, `Diesel`, `Electric`, `LPG` | **CNG / Other** |
| Transmission | `Manual` | **Automatic** |
| Seats | `Seats_5`, `Seats_Above_5` | **Fewer than 5** |

The serving layer (`car_pricing.predict`) reproduces these baselines exactly, so
"Dealer" and "CNG" are valid, unbiased inputs.

## Cleaning rules (`car_pricing.data.clean`)

- Rename `5` → `Seats_5`, `>5` → `Seats_Above_5`.
- Drop rows with missing or non-positive `selling_price`, then any remaining NA.
- Uppercase + strip `make` / `model` so the encoder sees canonical keys.

On this dataset the cleaning drops **0 rows** — it is already clean — but the
guard rails matter for future refreshes.

## Derived: price band

`selling_price` is bucketed into **Low / Medium / High** at the **terciles**
(33rd / 67th percentile) of the *training* set. On this split the edges are
**[0.3, 3.99, 6.75, 20.9] Lakhs**. At serve time the band is **derived from the
predicted price** using these edges (see [MODEL_CARD.md](MODEL_CARD.md)).
