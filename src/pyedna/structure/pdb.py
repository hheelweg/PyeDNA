"""Small PDB fixed-width formatting helpers for structure workflows."""


def set_chain_and_segid(line, chain="A", segid="A"):
    """Set fixed-width PDB chain and segment identifiers on one record."""

    line = line[:21] + chain + line[22:]
    return line.ljust(76)[:72] + f"{segid:>4s}" + line.ljust(76)[76:]
