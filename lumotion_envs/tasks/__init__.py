"""Task implementations — one module per task family, cataloged in
`lumotion_envs.registry`. Importing a task module bootstraps the engine
(`lumotion_envs._engine.ensure_engine`), so keep this package import-light:
anything that must work without the engine (--list, configs) lives outside.
"""
