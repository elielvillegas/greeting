"""Daily WhatsApp-ready prediction block.

One command: refresh results, fit the model, print a copy-paste block (EN/ES)
with 1st-half / 2nd-half / full-time probabilities for every match on a date.

  python -m src.whatsapp                      # today, English
  python -m src.whatsapp --lang es            # today, Spanish
  python -m src.whatsapp --date 2026-06-18 --lang both
  python -m src.whatsapp --no-refresh         # skip re-downloading data
"""
from __future__ import annotations
import argparse, datetime

from .data import load_internationals, load_worldcup
from .fixtures import predict_date, HOST_NATIONS

ES_TEAMS = {
    "Algeria": "Argelia", "Belgium": "Bélgica", "Bosnia and Herzegovina": "Bosnia",
    "Brazil": "Brasil", "Canada": "Canadá", "Cape Verde": "Cabo Verde",
    "Croatia": "Croacia", "Curaçao": "Curazao", "Czech Republic": "Chequia",
    "DR Congo": "RD Congo", "Egypt": "Egipto", "England": "Inglaterra",
    "France": "Francia", "Germany": "Alemania", "Haiti": "Haití", "Iran": "Irán",
    "Iraq": "Irak", "Ivory Coast": "Costa de Marfil", "Japan": "Japón",
    "Jordan": "Jordania", "Mexico": "México", "Morocco": "Marruecos",
    "Netherlands": "Holanda", "New Zealand": "Nueva Zelanda", "Norway": "Noruega",
    "Panama": "Panamá", "Qatar": "Catar", "Saudi Arabia": "Arabia Saudita",
    "Scotland": "Escocia", "South Africa": "Sudáfrica", "South Korea": "Corea del Sur",
    "Spain": "España", "Sweden": "Suecia", "Switzerland": "Suiza", "Tunisia": "Túnez",
    "Turkey": "Turquía", "United States": "Estados Unidos", "Uzbekistan": "Uzbekistán",
}
ES_CITY = {"Philadelphia": "Filadelfia", "Mexico City": "Ciudad de México",
           "New York": "Nueva York", "Los Angeles": "Los Ángeles"}
ES_DAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
ES_MON = ["", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def _city(ground, lang):
    c = (ground or "").split(" (")[0]
    return ES_CITY.get(c, c) if lang == "es" else c


def _pick_line(res, t1, t2, lang):
    ft = res["FT"]
    fav, other, fp = (t1, t2, ft["p1"]) if ft["p1"] >= ft["p2"] else (t2, t1, ft["p2"])
    i, j, _ = ft["ml"]
    score = f"{i}-{j}"
    drawy = ft["pdraw"] >= 0.34
    if lang == "es":
        if fp >= 0.72:
            base = f"Gana {fav}"
        elif fp >= 0.52:
            base = f"{fav} favorito"
        elif ft["pdraw"] >= max(ft["p1"], ft["p2"]):
            base = f"Muy parejo, leve ventaja {fav}"
        else:
            base = f"{fav} ligero favorito"
        extra = " (huele a empate)" if drawy else ""
        return f"🟢 {base}{extra} — probable {score}"
    if fp >= 0.72:
        base = f"{fav} should win"
    elif fp >= 0.52:
        base = f"{fav} favorite"
    elif ft["pdraw"] >= max(ft["p1"], ft["p2"]):
        base = f"Coin flip, slight edge {fav}"
    else:
        base = f"{fav} slight favorite"
    extra = " (draw very possible)" if drawy else ""
    return f"🟢 {base}{extra} — likely {score}"


def _nm(team, lang):
    return ES_TEAMS.get(team, team) if lang == "es" else team


def _row(label, a, res, ph, b, lang):
    r = res[ph]
    draw = "empate" if lang == "es" else "draw"
    return f"{label:11}{a} {r['p1']*100:.0f}% | {draw} {r['pdraw']*100:.0f}% | {b} {r['p2']*100:.0f}%"


def format_block(blocks, date, lang):
    d = datetime.date.fromisoformat(date)
    hosts_play = any(b[2] in HOST_NATIONS for b in blocks)
    out = []
    if lang == "es":
        out.append(f"🏆 PREDICCIONES MUNDIAL 2026 — {ES_DAYS[d.weekday()]} {d.day} {ES_MON[d.month]}")
        out.append("(modelo estadístico, no brujería 😅)")
        labels = ("1er tiempo:", "2do tiempo:", "FINAL:")
    else:
        out.append(f"🏆 WORLD CUP 2026 PREDICTIONS — {d.strftime('%A, %b %d')}")
        out.append("(statistical model, not witchcraft 😅)")
        labels = ("1st half:", "2nd half:", "FULL TIME:")

    for t1, t2, host, neutral, res, ground, rnd, note in blocks:
        a, b = _nm(t1, lang), _nm(t2, lang)
        out.append("")
        tag = "" if neutral else (" 🏠" if host == t1 else "")
        tag2 = " 🏠" if host == t2 else ""
        out.append(f"⚽ {a.upper()}{tag} vs {b.upper()}{tag2} — {_city(ground, lang)}")
        out.append(_row(labels[0], a, res, "1H", b, lang))
        out.append(_row(labels[1], a, res, "2H", b, lang))
        out.append(_row(labels[2], a, res, "FT", b, lang))
        out.append(_pick_line(res, a, b, lang))

    out.append("")
    if lang == "es":
        venue = ("Hoy todos juegan en cancha neutral (ningún anfitrión)."
                 if not hosts_play else "El anfitrión (🏠) juega de local y tiene ventaja.")
        out.append("📌 CÓMO FUNCIONA (en corto):")
        out.append(
            "El modelo agarra TODOS los partidos de selecciones de la historia, pero le da "
            "más peso a lo reciente y a los torneos grandes: el Mundial 2026 pesa muchísimo, "
            "el 2024-25 bastante, el Mundial 2022 poquito y lo de antes casi nada. Calcula la "
            "fuerza de ataque y defensa de cada selección (estilo Poisson/Dixon-Coles) y de ahí "
            "saca los goles esperados y las probabilidades. Solo dejamos las variables comprobadas "
            "estadísticamente: fuerza del equipo, localía y la diferencia entre 1er y 2do tiempo "
            "(se mete más gol en el segundo). Probamos altura, calor y hasta el cooling break, y "
            f"NINGUNO mueve los goles de forma significativa, así que fuera. {venue} Ojo: son "
            "probabilidades, no certezas — el fútbol es fútbol 🤷‍♂️⚽")
    else:
        venue = ("Today everyone plays on a neutral field (no host nations)."
                 if not hosts_play else "The host nation (🏠) plays at home and gets a boost.")
        out.append("📌 HOW IT WORKS (short version):")
        out.append(
            "The model takes EVERY international match in history, but weights recent games and "
            "big tournaments way more: World Cup 2026 counts a ton, 2024-25 a good amount, World "
            "Cup 2022 a little, older stuff almost nothing. It rates each team's attack and defense "
            "(Poisson/Dixon-Coles style) and from that gets expected goals and probabilities. We "
            "kept only the statistically proven variables: team strength, home advantage, and the "
            "1st-vs-2nd half difference (more goals come in the 2nd). We tested altitude, heat, and "
            f"even the cooling break — none move scoring significantly, so we dropped them. {venue} "
            "Heads up: these are probabilities, not guarantees — football's gonna football 🤷‍♂️⚽")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--lang", choices=["en", "es", "both"], default="en")
    ap.add_argument("--no-refresh", action="store_true", help="skip re-downloading results")
    args = ap.parse_args()

    if not args.no_refresh:
        load_internationals(refresh=True)
        load_worldcup(refresh=True)

    blocks = predict_date(args.date)
    if not blocks:
        print(f"No fixtures with confirmed teams on {args.date}.")
        return
    langs = ["en", "es"] if args.lang == "both" else [args.lang]
    for i, lang in enumerate(langs):
        if i:
            print("\n" + "=" * 60 + "\n")
        print(format_block(blocks, args.date, lang))


if __name__ == "__main__":
    main()
