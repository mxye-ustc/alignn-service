"""掺杂相关 API 路由"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from alignn_service.core.doping_generator import DopingGenerator, DopingError
from alignn_service.utils.file_parser import FileParser, FileParseError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/doping", tags=["掺杂"])


@router.post("/generate")
async def generate_doping_config(
    host_structure_content: str,
    dopant: str,
    doping_site: str,
    n_dopant: int,
    concentration: Optional[float] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    生成单个掺杂构型

    Args:
        host_structure_content: 宿主结构文件内容
        dopant: 掺杂元素
        doping_site: 掺杂位点
        n_dopant: 掺杂原子数
        concentration: 浓度百分比
        seed: 随机种子
    """
    try:
        atoms = FileParser.parse_content(host_structure_content)

        generator = DopingGenerator()
        doped = generator.generate_random_doping(
            host_atoms=atoms,
            dopant=dopant,
            doping_site=doping_site,
            n_dopant=n_dopant,
            seed=seed
        )

        config_id = f"{atoms.composition.reduced_formula}_{dopant}_{n_dopant}"

        config = generator.save_config(
            atoms=doped,
            config_id=config_id,
            dopant=dopant,
            doping_site=doping_site,
            n_dopant=n_dopant,
            concentration=concentration
        )

        return {
            "status": "success",
            "config": config
        }

    except (FileParseError, DopingError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"掺杂生成失败: {e}")
        raise HTTPException(status_code=500, detail="掺杂生成失败")


@router.post("/generate/batch")
async def generate_batch_configs(
    host_structure_content: str,
    dopants: List[str],
    doping_site: str,
    concentrations: List[float],
    n_configs_per_combination: int = 3
) -> Dict[str, Any]:
    """
    批量生成掺杂构型

    Args:
        host_structure_content: 宿主结构文件内容
        dopants: 掺杂元素列表
        doping_site: 掺杂位点
        concentrations: 浓度列表
        n_configs_per_combination: 每个组合的构型数
    """
    try:
        atoms = FileParser.parse_content(host_structure_content)

        generator = DopingGenerator()
        configs = generator.generate_batch(
            host_atoms=atoms,
            dopants=dopants,
            doping_site=doping_site,
            concentrations=concentrations,
            n_configs_per_combination=n_configs_per_combination,
            save=True
        )

        return {
            "status": "success",
            "total_configs": len(configs),
            "configs": configs
        }

    except (FileParseError, DopingError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"批量掺杂生成失败: {e}")
        raise HTTPException(status_code=500, detail="批量掺杂生成失败")


@router.get("/sites/{formula}")
async def get_doping_sites(formula: str) -> Dict[str, Any]:
    """
    获取指定化合物的可用掺杂位点

    Args:
        formula: 化学式
    """
    # 预定义的位点信息
    site_info = {
        "LiFePO4": {
            "Li": {"count": 4, "wyckoff": "4a", "description": "锂位"},
            "Fe": {"count": 4, "wyckoff": "4c", "description": "铁位"},
            "P": {"count": 4, "wyckoff": "4c", "description": "磷位"},
        },
        "LiCoO2": {
            "Li": {"count": 1, "wyckoff": "3a", "description": "锂位"},
            "Co": {"count": 1, "wyckoff": "3a", "description": "钴位"},
        },
        "LiMn2O4": {
            "Li": {"count": 1, "wyckoff": "8a", "description": "锂位"},
            "Mn": {"count": 2, "wyckoff": "16d", "description": "锰位"},
        },
    }

    info = site_info.get(formula, {})

    return {
        "formula": formula,
        "sites": info
    }
