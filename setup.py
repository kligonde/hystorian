import setuptools
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setuptools.setup(
    version_config={
        "template": "{tag}",
        "dev_template": "{tag}.dev{ccount}",
        "dirty_template": "{tag}.dev{ccount}",
        "starting_version": "0.0.1",
        "version_callback": None,
        "version_file": None,
        "count_commits_from_version_file": True,
    },
    setup_requires=["setuptools-git-versioning"],
    name="hystorian",
    author="Loïc Musy <loic.musy@unige.ch>, Ralph Bulanadi <ralph.bulanadi@unige.ch>",
    author_email="loic.musy@unige.ch",
    description="a generic materials science data analysis Python package built with processing traceability, reproducibility, and archival ability at its core.",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://gitlab.unige.ch/paruch-group/hystorian",
    packages=["hystorian", "hystorian.io", "hystorian.processing"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    python_requires=">=3.10",
)
