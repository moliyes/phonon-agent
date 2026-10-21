from pathlib import Path

import yaml


def test_skill_frontmatter():
    root = Path(__file__).parents[1] / "skills" / "materials-science"
    names = set()
    for path in root.glob("*/SKILL.md"):
        text = path.read_text()
        assert text.startswith("---\n")
        _, frontmatter, _ = text.split("---", 2)
        data = yaml.safe_load(frontmatter)
        assert data["description"].endswith(".")
        assert len(data["description"]) <= 60
        platforms = data.get("platforms", data["metadata"]["hermes"].get("platforms"))
        assert platforms == ["linux"]
        names.add(data["name"])
    assert names == {
        "phonon-agent",
        "dpa4-md-sampling",
        "vasp-labeling",
        "dpa4-finetuning",
    }
