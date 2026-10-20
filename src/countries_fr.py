"""French country names (L'Équipe / France Football juror lists) -> canonical names of countries.csv.

Used by parse_bdo.py and companion_bdo.py; `python -I nhb/src/countries_fr.py` checks that every target is a
canonical name in nhb/data/processed/countries.csv and prints any that is not.

L'Équipe prints the juror's country in French, in mixed case (2022, women's lists) or capitals
(men 2025), with or without accents ("ISRAEL", "HAITI", "COTE D'IVOIRE"). Names are compared after
countries.fold() (accents, case and punctuation removed), so one entry covers every variant. Each
entry was written by hand; nothing is translated automatically. One typo on the women's 2025 page
("Brésll" for Brésil) is listed explicitly. "Eire" and "Irlande" both mean the Republic of Ireland
(Northern Ireland is "Irlande du Nord" on the same lists).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from countries import fold  # noqa: E402

FR = {
    "Afrique du Sud": "South Africa", "Albanie": "Albania", "Algérie": "Algeria", "Allemagne": "Germany",
    "Andorre": "Andorra", "Angleterre": "England", "Angola": "Angola", "Arabie saoudite": "Saudi Arabia",
    "Argentine": "Argentina", "Arménie": "Armenia", "Aruba": "Aruba", "Australie": "Australia",
    "Autriche": "Austria", "Azerbaïdjan": "Azerbaijan", "Bahreïn": "Bahrain", "Barbade": "Barbados",
    "Belgique": "Belgium", "Belize": "Belize", "Bénin": "Benin", "Bermudes": "Bermuda", "Bhoutan": "Bhutan",
    "Biélorussie": "Belarus", "Bolivie": "Bolivia", "Bosnie-Herzégovine": "Bosnia and Herzegovina",
    "Brésil": "Brazil", "Brésll": "Brazil",   # typo on the women's 2025 page
    "Bulgarie": "Bulgaria", "Burkina Faso": "Burkina Faso", "Burundi": "Burundi", "Cambodge": "Cambodia",
    "Cameroun": "Cameroon", "Canada": "Canada", "Cap-Vert": "Cabo Verde", "Chili": "Chile", "Chine": "China PR",
    "Chypre": "Cyprus", "Colombie": "Colombia", "Congo": "Congo", "RD Congo": "Congo DR",
    "Corée du Sud": "Korea Republic", "Costa Rica": "Costa Rica", "Côte d'Ivoire": "Côte d'Ivoire",
    "Croatie": "Croatia", "Cuba": "Cuba", "Curaçao": "Curaçao", "Danemark": "Denmark", "Djibouti": "Djibouti",
    "Écosse": "Scotland", "Égypte": "Egypt", "Eire": "Republic of Ireland", "Irlande": "Republic of Ireland",
    "Émirats arabes unis": "United Arab Emirates", "Équateur": "Ecuador", "Érythrée": "Eritrea",
    "Espagne": "Spain", "Estonie": "Estonia", "Eswatini": "Eswatini", "États-Unis": "United States",
    "Éthiopie": "Ethiopia", "Féroé": "Faroe Islands", "Finlande": "Finland", "France": "France", "Gabon": "Gabon",
    "Galles": "Wales", "Gambie": "The Gambia", "Géorgie": "Georgia", "Ghana": "Ghana", "Grèce": "Greece",
    "Grenade": "Grenada", "Guatemala": "Guatemala", "Guinée": "Guinea", "Guinée-Bissau": "Guinea-Bissau",
    "Guinée équatoriale": "Equatorial Guinea", "Haïti": "Haiti", "Honduras": "Honduras",
    "Hongkong": "Hong Kong, China", "Hong Kong": "Hong Kong, China", "Hongrie": "Hungary",
    "Îles Caïmans": "Cayman Islands", "Îles Cook": "Cook Islands", "Inde": "India", "Indonésie": "Indonesia",
    "Irak": "Iraq", "Iran": "IR Iran", "Irlande du Nord": "Northern Ireland", "Islande": "Iceland",
    "Israël": "Israel", "Italie": "Italy", "Jamaïque": "Jamaica", "Japon": "Japan", "Jordanie": "Jordan",
    "Kazakhstan": "Kazakhstan", "Kenya": "Kenya", "Kirghizistan": "Kyrgyz Republic", "Kosovo": "Kosovo",
    "Koweït": "Kuwait", "Liban": "Lebanon", "Liberia": "Liberia", "Libye": "Libya",
    "Liechtenstein": "Liechtenstein", "Lituanie": "Lithuania", "Luxembourg": "Luxembourg", "Macao": "Macau",
    "Macédoine du Nord": "North Macedonia", "Madagascar": "Madagascar", "Malaisie": "Malaysia",
    "Malawi": "Malawi", "Maldives": "Maldives", "Mali": "Mali", "Malte": "Malta", "Maroc": "Morocco",
    "Maurice": "Mauritius", "Mauritanie": "Mauritania", "Mexique": "Mexico", "Moldavie": "Moldova",
    "Mongolie": "Mongolia", "Monténégro": "Montenegro", "Mozambique": "Mozambique", "Namibie": "Namibia",
    "Népal": "Nepal", "Nicaragua": "Nicaragua", "Niger": "Niger", "Nigeria": "Nigeria", "Norvège": "Norway",
    "Nouvelle-Calédonie": "New Caledonia", "Nouvelle-Zélande": "New Zealand", "Oman": "Oman",
    "Ouganda": "Uganda", "Ouzbékistan": "Uzbekistan", "Pakistan": "Pakistan", "Palestine": "Palestine",
    "Panama": "Panama", "Paraguay": "Paraguay", "Pays-Bas": "Netherlands", "Pérou": "Peru",
    "Philippines": "Philippines", "Pologne": "Poland", "Porto Rico": "Puerto Rico", "Portugal": "Portugal",
    "Qatar": "Qatar", "République centrafricaine": "Central African Republic",
    "République dominicaine": "Dominican Republic", "République tchèque": "Czechia", "Roumanie": "Romania",
    "Russie": "Russia", "Rwanda": "Rwanda", "Saint-Marin": "San Marino", "Salvador": "El Salvador",
    "Sao Tomé-et-Principe": "São Tomé and Príncipe", "Sénégal": "Senegal", "Serbie": "Serbia",
    "Seychelles": "Seychelles", "Sierra Leone": "Sierra Leone", "Singapour": "Singapore",
    "Slovaquie": "Slovakia", "Slovénie": "Slovenia", "Soudan": "Sudan", "Soudan du Sud": "South Sudan",
    "Sri Lanka": "Sri Lanka", "Suède": "Sweden", "Suisse": "Switzerland", "Suriname": "Suriname",
    "Syrie": "Syria", "Tadjikistan": "Tajikistan", "Tahiti": "Tahiti", "Taïwan": "Chinese Taipei",
    "Tanzanie": "Tanzania", "Tchad": "Chad", "Thaïlande": "Thailand", "Togo": "Togo",
    "Trinité-et-Tobago": "Trinidad and Tobago", "Tunisie": "Tunisia", "Turkménistan": "Turkmenistan",
    "Turks-et-Caïcos": "Turks and Caicos Islands", "Turquie": "Türkiye", "Ukraine": "Ukraine",
    "Uruguay": "Uruguay", "Venezuela": "Venezuela", "Vietnam": "Vietnam", "Yémen": "Yemen",
    "Zambie": "Zambia", "Zimbabwe": "Zimbabwe",
}
BY_FOLD = {fold(k): v for k, v in FR.items()}
MAX_WORDS = max(len(k.split()) for k in BY_FOLD)


def resolve_fr(name):
    """Canonical country for a French name, or None."""
    return BY_FOLD.get(fold(name))


def split_country(text):
    """Split 'COUNTRY rest' at the longest leading run of words that is a known French country name.
    Returns (country_raw, rest) or (None, text)."""
    words = text.split()
    for n in range(min(MAX_WORDS, len(words)), 0, -1):
        head = " ".join(words[:n])
        if fold(head) in BY_FOLD:
            return head.strip(" -.,:"), " ".join(words[n:])
    return None, text


def main():
    import pandas as pd
    canon = set(pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "processed" / "countries.csv",
                            keep_default_na=False).country)
    bad = sorted({v for v in FR.values() if v not in canon})
    print(f"{len(FR)} French names -> {len(set(FR.values()))} countries; targets not canonical: {bad}",
          file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
