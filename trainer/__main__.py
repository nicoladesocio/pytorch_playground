if __package__ in (None, ""):
    from trainer import main
else:
    from .trainer import main


if __name__ == "__main__":
    raise SystemExit(main())
