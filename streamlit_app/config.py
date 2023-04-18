# Authors:
# Thomas Dumazert
# Hajer Souaifi-Amara
# Creation date: 23FEB2023 by Thomas
# Modification date: 21MAR2023 by Hajer
# Modification date: 23MAR2023 by Thomas
# Modification date: 27MAR2023 by Hajer
"""

Config file for Streamlit App

"""

from member import Member


TITLE = "MyeLeuNet"

TEAM_MEMBERS = [
    Member(
        name="Thomas DUMAZERT",
        linkedin_url="https://www.linkedin.com/in/dumazertthomas/",
        github_url="https://github.com/ThomasDumazert",
    ),
    Member(
        name="Etienne MASFARAUD",
        linkedin_url="https://www.linkedin.com/in/e-masfaraud/",
        github_url="https://github.com/Emasfa",
    ),
    Member(
        name="Hajer SOUAIFI-AMARA",
        linkedin_url="https://www.linkedin.com/in/hajersouaifiamara/",
        github_url="https://github.com/HAJAMARA",
    ),
    #Member("Gwenaëlle WAYARIDRI"),
]

PROMOTION = "Bootcamp Data Scientist - Promotion Janvier 2023"
