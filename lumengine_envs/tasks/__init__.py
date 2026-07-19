"""Task implementations — one module per task family, cataloged in
`lumengine_envs.registry`. Importing a task module bootstraps the engine
(`lumengine_envs._engine.ensure_engine`), so keep this package import-light:
anything that must work without the engine (--list, configs) lives outside.
"""
