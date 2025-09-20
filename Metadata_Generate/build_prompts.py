import pandas as pd
import numpy as np
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python build_prompts.py <input_file.csv/xlsx>")
        sys.exit(1)

    in_path = Path(sys.argv[1])

    # Load dataset
    if in_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(in_path)
    else:
        df = pd.read_csv(in_path)

    # Known genre columns
    KNOWN_GENRES = [
        "Action","Adventure","Animation","Biography","Comedy","Crime","Documentary","Drama",
        "Family","Fantasy","History","Horror","Music","Musical","Mystery","Romance","Sci-Fi",
        "Sport","Thriller","War","Western"
    ]
    present_genres = [g for g in KNOWN_GENRES if g in df.columns]

    def clean_text(x):
        if pd.isna(x): return ""
        return " ".join(str(x).split())

    def fmt_budget(v):
        try:
            if pd.isna(v): return None
            v = float(v)
            if not np.isfinite(v): return None
            if abs(v) >= 1e6:
                return f"${v/1e6:.1f}M"
            elif abs(v) >= 1e3:
                return f"${v/1e3:.0f}K"
            else:
                return f"${v:.0f}"
        except Exception:
            return None

    def genres_from_row(row):
        genres = []
        for g in present_genres:
            try:
                if float(row.get(g, 0)) == 1:
                    genres.append(g)
            except Exception:
                pass
        return ", ".join(genres)

    def first_n_nonempty(vals, n):
        out = [clean_text(v) for v in vals if clean_text(v)]
        return out[:n]

    def build_prompt(row):
        title = clean_text(row.get("Series_Title", ""))
        director = clean_text(row.get("Director", ""))
        year = clean_text(row.get("release_year", ""))
        imdb = clean_text(row.get("IMDB_Rating", ""))
        budget = fmt_budget(row.get("budget", None))
        stars = first_n_nonempty(
            [row.get("Star1",""), row.get("Star2",""), row.get("Star3",""), row.get("Star4","")],
            4
        )
        genres = genres_from_row(row)

        parts = []
        if genres:
            parts.append(f"A {genres} film")
        else:
            parts.append("A feature film")

        if year:
            parts[-1] += f" released in {year}"
        if director:
            parts.append(f"directed by {director}")
        if stars:
            if len(stars) == 1:
                parts.append(f"starring {stars[0]}")
            elif len(stars) == 2:
                parts.append(f"starring {stars[0]} and {stars[1]}")
            else:
                parts.append(f"starring {', '.join(stars[:-1])}, and {stars[-1]}")
        if budget:
            parts.append(f"with a budget of {budget}")
        if imdb:
            parts.append(f"(IMDb rating: {imdb})")

        text = "; ".join(parts) + "."
        if title:
            text = f"'{title}': " + text
        return text

    # Add new column
    df["meta_prompt"] = df.apply(build_prompt, axis=1)

    # Save back to the same file
    if in_path.suffix.lower() in [".xlsx", ".xls"]:
        df.to_excel(in_path, index=False)
    else:
        df.to_csv(in_path, index=False)

    print(f"Updated file saved in place: {in_path}")

if __name__ == "__main__":
    main()