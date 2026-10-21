from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile, is_zipfile

import yaml

from .errors import ConfigError


PRIVATE_CONFIG_GUIDANCE = (
    "尚未配置用户级 Bohrium/POTCAR 信息。\n"
    "请在本机终端运行：phonon-agent setup\n"
    "不要在 Hermes 对话中发送密码或 POTCAR 内容。\n"
    "完成后告诉我“配置完成”，我会重新验证。"
)

DEEPMD_ENV_GUIDANCE = (
    "尚未配置可用的 DeepMD-kit 计算环境。\n"
    "请先自行安装 DeepMD-kit，然后运行：phonon-agent setup\n"
    "配置时填写环境目录或该环境的 Python 路径。"
)

DEFAULT_MACHINE_TYPE = "c16_m64_cpu"
DEFAULT_IMAGE = "registry.dp.tech/dptech/prod-604/vasp:6"
DEFAULT_CORES = 16
DEFAULT_COMMAND_TEMPLATE = "mpirun -n {cores} vasp_std"


def config_dir() -> Path:
    override = os.environ.get("PHONON_AGENT_CONFIG_DIR")
    return Path(override).expanduser().resolve() if override else Path("~/.config/phonon-agent").expanduser()


@dataclass(frozen=True)
class UserConfig:
    path: Path
    exists: bool
    potcar_source: Path | None
    potcar_family: str
    machine: Path
    resources: Path
    cores: int
    command_template: str
    deepmd_python: Path | None
    ph_command: str


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_user_config() -> UserConfig:
    root = config_dir()
    path = root / "config.yaml"
    raw: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ConfigError(f"user config must be a mapping: {path}")
        if int(loaded.get("schema_version", 0)) != 1:
            raise ConfigError(f"user config schema_version must be 1: {path}")
        raw = loaded
    potcar = dict(raw.get("potcar") or {})
    source_value = potcar.get("source")
    bohrium = dict(raw.get("bohrium") or {})
    environments = dict(raw.get("environments") or {})
    deepmd = dict(environments.get("deepmd") or {})
    deepmd_value = deepmd.get("python")
    tools = dict(raw.get("tools") or {})
    phonon_kit = dict(tools.get("phonon_kit") or {})
    cores = int(bohrium.get("cores", DEFAULT_CORES))
    if cores <= 0:
        raise ConfigError("user config bohrium.cores must be positive")
    template = str(bohrium.get("command_template", DEFAULT_COMMAND_TEMPLATE))
    try:
        template.format(cores=cores)
    except (KeyError, ValueError) as exc:
        raise ConfigError("bohrium.command_template may only require {cores}") from exc
    return UserConfig(
        path=path,
        exists=path.is_file(),
        potcar_source=_resolve(root, source_value) if source_value else None,
        potcar_family=str(potcar.get("family", "")).strip("/"),
        machine=_resolve(root, bohrium.get("machine", "bohrium/machine.json")),
        resources=_resolve(root, bohrium.get("resources", "bohrium/resources.json")),
        cores=cores,
        command_template=template,
        deepmd_python=_resolve(root, deepmd_value) if deepmd_value else None,
        ph_command=str(phonon_kit.get("command", Path("~/.local/bin/ph").expanduser())),
    )


def _atomic_private_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.bak-{timestamp}")
    counter = 1
    while destination.exists():
        destination = path.with_name(f"{path.name}.bak-{timestamp}-{counter}")
        counter += 1
    shutil.copy2(path, destination)
    destination.chmod(0o600)
    return destination


def _prompt_required(prompt: str, *, secret: bool = False) -> str:
    reader = getpass.getpass if secret else input
    while True:
        value = reader(prompt).strip()
        if value:
            return value
        print("该项不能为空。")


def _prompt_secret_twice(label: str) -> str:
    while True:
        first = _prompt_required(f"{label}（隐藏输入）: ", secret=True)
        second = _prompt_required(f"再次输入{label}: ", secret=True)
        if first == second:
            return first
        print("两次输入不一致，请重试。")


def _prompt_positive_int(prompt: str) -> int:
    while True:
        value = _prompt_required(prompt)
        try:
            result = int(value)
        except ValueError:
            result = 0
        if result > 0:
            return result
        print("请输入正整数。")


def normalize_python_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "bin" / "python"
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ConfigError(f"DeepMD Python is not executable: {path}")
    return path


def probe_python_module(python: Path, module: str) -> tuple[bool, str]:
    code = (
        "import importlib; "
        f"m=importlib.import_module({module!r}); "
        "print(getattr(m, '__version__', 'installed'))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", code], text=True, capture_output=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, type(exc).__name__
    if result.returncode:
        return False, "import failed"
    return True, result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "installed"


def _project_roots() -> tuple[Path | None, Path | None]:
    agent_root = Path(__file__).resolve().parents[2]
    kit_root = agent_root.parent / "phonon-kit"
    return (
        agent_root if (agent_root / "pyproject.toml").is_file() else None,
        kit_root if (kit_root / "pyproject.toml").is_file() else None,
    )


def _install_package(python: Path, requirement: str) -> None:
    result = subprocess.run(
        [str(python), "-m", "pip", "install", requirement],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-12:])
        raise ConfigError(f"automatic tool installation failed for {requirement}:\n{tail}")


def _global_bin_dir() -> Path:
    override = os.environ.get("PHONON_AGENT_BIN_DIR")
    return Path(override).expanduser().resolve() if override else Path("~/.local/bin").expanduser()


def _link_global_command(name: str, target: Path) -> Path:
    destination = _global_bin_dir() / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_symlink():
        if destination.samefile(target):
            return destination
        raise ConfigError(f"refusing to replace non-symlink command: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def configure_compute_environment(value: str | Path) -> dict[str, Any]:
    python = normalize_python_path(value)
    deepmd_ok, deepmd_version = probe_python_module(python, "deepmd")
    if not deepmd_ok:
        raise ConfigError(f"DeepMD-kit is not importable with {python}; phonon-agent will not install it")
    agent_root, kit_root = _project_roots()
    agent_executable = python.parent / "phonon-agent"
    ph_executable = python.parent / "ph"
    required = ("phonon_agent", "quests", "dpdata")
    if not agent_executable.is_file() or any(not probe_python_module(python, item)[0] for item in required):
        requirement = (
            f"{agent_root}[sampling,conversion]"
            if agent_root
            else "phonon-agent[sampling,conversion] @ git+https://github.com/moliyes/phonon-agent.git"
        )
        _install_package(python, requirement)
    if not ph_executable.is_file() or not probe_python_module(python, "phonon_kit")[0]:
        _install_package(python, str(kit_root) if kit_root else "dpa-phonon")
    if not agent_executable.is_file() or not ph_executable.is_file():
        raise ConfigError("tool installation completed without phonon-agent/ph console commands")
    agent_command = _link_global_command("phonon-agent", agent_executable)
    ph_command = _link_global_command("ph", ph_executable)
    return {
        "python": python,
        "deepmd_version": deepmd_version,
        "phonon_agent_command": agent_command,
        "ph_command": ph_command,
    }


def _family_candidates(source: Path) -> list[str]:
    candidates: set[str] = set()
    if source.is_dir():
        for path in source.rglob("POTCAR"):
            try:
                relative = path.relative_to(source)
            except ValueError:
                continue
            if len(relative.parts) >= 2:
                candidates.add(PurePosixPath(*relative.parts[:-2]).as_posix().strip("."))
    elif source.is_file() and is_zipfile(source):
        try:
            with ZipFile(source) as archive:
                for name in archive.namelist():
                    parts = PurePosixPath(name).parts
                    if len(parts) >= 2 and parts[-1] == "POTCAR":
                        candidates.add(PurePosixPath(*parts[:-2]).as_posix().strip("."))
        except BadZipFile:
            return []
    return sorted(item for item in candidates if item not in {"", "."})


def _choose_family(source: Path) -> str:
    candidates = _family_candidates(source)
    if len(candidates) == 1:
        print(f"自动识别 POTCAR family: {candidates[0]}")
        return candidates[0]
    if candidates:
        print("检测到多个 POTCAR family：")
        for index, value in enumerate(candidates, start=1):
            print(f"  {index}. {value}")
        while True:
            choice = input("请选择编号，或直接输入相对 family 路径: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                return candidates[int(choice) - 1]
            if choice:
                return choice.strip("/")
    return input("POTCAR family 相对路径（若 source 直接包含元素目录可留空）: ").strip().strip("/")


def _legacy_machine(email: str, password: str, project_id: int) -> dict[str, Any]:
    return {
        "batch_type": "Bohrium",
        "context_type": "BohriumContext",
        "local_root": "./",
        "remote_profile": {
            "email": email,
            "password": password,
            "project_id": project_id,
            "input_data": {
                "job_type": "container",
                "grouped": "True",
                "log_file": "log",
                "job_name": "phonon-agent",
                "scass_type": DEFAULT_MACHINE_TYPE,
                "platform": "ali",
                "image_name": DEFAULT_IMAGE,
                "on_demand": 0,
            },
        },
    }


def _openapi_machine(
    project_id: int | None, access_key: str | None, app_key: str | None
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "machine_type": DEFAULT_MACHINE_TYPE,
        "platform": "ali",
        "image_address": DEFAULT_IMAGE,
        "job_name": "phonon-agent",
    }
    if project_id is not None:
        profile["project_id"] = project_id
    if access_key:
        profile["access_key"] = access_key
    if app_key:
        profile["app_key"] = app_key
    return {
        "batch_type": "OpenAPI",
        "context_type": "OpenAPIContext",
        "local_root": "./",
        "remote_profile": profile,
    }


def run_setup() -> dict[str, Any]:
    root = config_dir()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    print(f"phonon-agent 私有配置目录：{root}")
    print("秘密只写入本机权限为 600 的文件；不要把密码或 POTCAR 内容发送到聊天中。")
    compute = configure_compute_environment(
        _prompt_required("DeepMD-kit 环境目录或 Python 路径: ")
    )
    print(f"DeepMD-kit {compute['deepmd_version']}：{compute['python']}")
    print(f"phonon-kit 全局命令：{compute['ph_command']}")
    print("建议随后运行 `ph model freeze`，在本机冻结 builtin:dpa4；未冻结时会回退到包内 .pt。")
    print("认证方式：1=邮箱/密码，2=OpenAPI 密钥文件，3=OpenAPI 环境变量")
    while True:
        auth = input("请选择 [1]: ").strip() or "1"
        if auth in {"1", "2", "3"}:
            break
        print("请输入 1、2 或 3。")
    if auth == "1":
        email = _prompt_required("Bohrium 邮箱: ")
        password = _prompt_secret_twice("Bohrium 密码")
        project_id = _prompt_positive_int("Bohrium project ID: ")
        machine = _legacy_machine(email, password, project_id)
        auth_mode = "legacy-password"
    elif auth == "2":
        access_key = _prompt_secret_twice("Bohrium Access Key")
        project_id = _prompt_positive_int("Bohrium project ID: ")
        app_key = getpass.getpass("Bohrium App Key（可留空）: ").strip() or None
        machine = _openapi_machine(project_id, access_key, app_key)
        auth_mode = "openapi-file"
    else:
        machine = _openapi_machine(None, None, None)
        auth_mode = "openapi-environment"
        print("提交前需设置 BOHRIUM_ACCESS_KEY 和 BOHRIUM_PROJECT_ID；BOHRIUM_APP_KEY 可选。")
    while True:
        source = Path(_prompt_required("POTCAR ZIP 或总目录路径: ")).expanduser().resolve()
        if source.is_dir() or (source.is_file() and is_zipfile(source)):
            break
        print("路径必须是已存在的目录或 ZIP 文件。")
    family = _choose_family(source)
    machine_path = root / "bohrium" / "machine.json"
    resources_path = root / "bohrium" / "resources.json"
    config_path = root / "config.yaml"
    backups = [item for path in (config_path, machine_path, resources_path) if (item := _backup(path))]
    public = {
        "schema_version": 1,
        "environments": {"deepmd": {"python": str(compute["python"])}},
        "tools": {"phonon_kit": {"command": str(compute["ph_command"])}},
        "potcar": {"source": str(source), "family": family},
        "bohrium": {
            "machine": "bohrium/machine.json",
            "resources": "bohrium/resources.json",
            "cores": DEFAULT_CORES,
            "command_template": DEFAULT_COMMAND_TEMPLATE,
        },
    }
    _atomic_private_write(config_path, yaml.safe_dump(public, sort_keys=False, allow_unicode=True))
    _atomic_private_write(machine_path, json.dumps(machine, ensure_ascii=False, indent=2) + "\n")
    _atomic_private_write(resources_path, json.dumps({"group_size": 1}, indent=2) + "\n")
    return {
        "configured": True,
        "config_dir": str(root),
        "auth_mode": auth_mode,
        "deepmd_python": str(compute["python"]),
        "deepmd_version": compute["deepmd_version"],
        "phonon_kit_command": str(compute["ph_command"]),
        "potcar_source_type": "directory" if source.is_dir() else "zip",
        "potcar_family": family,
        "cores": DEFAULT_CORES,
        "command": DEFAULT_COMMAND_TEMPLATE.format(cores=DEFAULT_CORES),
        "backups_created": [str(path) for path in backups],
    }


def _mode(path: Path) -> int | None:
    return stat.S_IMODE(path.stat().st_mode) if path.exists() else None


def machine_core_count(machine: dict[str, Any]) -> int | None:
    profile = machine.get("remote_profile") or {}
    input_data = profile.get("input_data") or {}
    machine_type = str(profile.get("machine_type") or input_data.get("scass_type") or "")
    match = re.search(r"(?:^|_)c(\d+)(?:_|$)", machine_type)
    return int(match.group(1)) if match else None


def command_core_count(command: str) -> int | None:
    match = re.search(r"(?:^|\s)(?:mpirun|mpiexec)(?:\s+[^\s]+)*?\s+-n\s+(\d+)(?:\s|$)", command)
    return int(match.group(1)) if match else None


def dynamic_job_name(project: str, run_name: str, maximum: int = 63) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", f"pa-{project}-{run_name}-vasp").strip("-")
    if len(base) <= maximum:
        return base
    suffix = hashlib.sha256(base.encode()).hexdigest()[:8]
    return f"{base[: maximum - len(suffix) - 1].rstrip('-')}-{suffix}"


def load_machine_safely(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load Bohrium machine config: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Bohrium machine config must be an object: {path}")
    return data


def check_private_config() -> dict[str, Any]:
    config = load_user_config()
    errors: list[str] = []
    warnings: list[str] = []
    root = config_dir()
    if not config.exists:
        return {
            "ok": False,
            "configured": False,
            "config_dir": str(root),
            "authentication": "unconfigured",
            "potcar_source_type": "missing",
            "configured_cores": config.cores,
            "deepmd_environment": "missing",
            "errors": [PRIVATE_CONFIG_GUIDANCE],
            "warnings": [],
        }
    if root.exists() and _mode(root) != 0o700:
        warnings.append("private config directory permissions should be 0700")
    if config.deepmd_python is None:
        errors.append(DEEPMD_ENV_GUIDANCE)
        deepmd_version = None
    else:
        deepmd_ok, deepmd_version = probe_python_module(config.deepmd_python, "deepmd")
        if not deepmd_ok:
            errors.append(f"DeepMD-kit is not importable with configured Python: {config.deepmd_python}")
    ph_path = Path(config.ph_command).expanduser()
    if not ph_path.is_file() or not os.access(ph_path, os.X_OK):
        errors.append(f"phonon-kit command missing or not executable: {ph_path}; run phonon-agent setup")
    machine: dict[str, Any] = {}
    auth_mode = "unconfigured"
    if not config.machine.is_file():
        errors.append(f"Bohrium machine config missing: {config.machine}")
    else:
        if _mode(config.machine) != 0o600:
            errors.append("Bohrium machine config permissions must be 0600")
        machine = load_machine_safely(config.machine)
        profile = machine.get("remote_profile") or {}
        if profile.get("email") and profile.get("password"):
            auth_mode = "legacy-password"
        elif profile.get("access_key"):
            auth_mode = "openapi-file"
        elif os.environ.get("BOHRIUM_ACCESS_KEY") and os.environ.get("BOHRIUM_PROJECT_ID"):
            auth_mode = "openapi-environment"
        else:
            errors.append("Bohrium authentication is incomplete; run phonon-agent setup")
    if not config.resources.is_file():
        errors.append(f"Bohrium resources config missing: {config.resources}")
    elif _mode(config.resources) != 0o600:
        errors.append("Bohrium resources config permissions must be 0600")
    if config.potcar_source is None or not config.potcar_source.exists():
        errors.append("POTCAR source is not configured or does not exist; run phonon-agent setup")
        source_type = "missing"
    elif config.potcar_source.is_dir():
        source_type = "directory"
    elif is_zipfile(config.potcar_source):
        source_type = "zip"
    else:
        source_type = "invalid"
        errors.append("POTCAR source must be a ZIP archive or directory")
    command = config.command_template.format(cores=config.cores)
    machine_cores = machine_core_count(machine) if machine else None
    command_cores = command_core_count(command)
    if machine and machine_cores is None:
        errors.append("cannot determine core count from Bohrium machine type")
    if machine_cores is not None and machine_cores != config.cores:
        errors.append(f"Bohrium machine has {machine_cores} cores but config declares {config.cores}")
    if command_cores != config.cores:
        errors.append(f"VASP command core count does not match configured {config.cores} cores")
    return {
        "ok": not errors,
        "configured": config.exists,
        "config_dir": str(root),
        "directory_mode": f"{_mode(root):04o}" if _mode(root) is not None else None,
        "machine_configured": config.machine.is_file(),
        "machine_mode": f"{_mode(config.machine):04o}" if _mode(config.machine) is not None else None,
        "resources_configured": config.resources.is_file(),
        "resources_mode": f"{_mode(config.resources):04o}" if _mode(config.resources) is not None else None,
        "authentication": auth_mode,
        "potcar_source_type": source_type,
        "potcar_family": config.potcar_family,
        "deepmd_python": str(config.deepmd_python) if config.deepmd_python else None,
        "deepmd_version": deepmd_version,
        "phonon_kit_command": str(ph_path),
        "configured_cores": config.cores,
        "machine_cores": machine_cores,
        "command_cores": command_cores,
        "errors": errors,
        "warnings": warnings,
    }
