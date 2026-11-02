"""Educational Schnorr-style zero-knowledge authentication."""
import hashlib, secrets
P, Q, G = 227, 113, 3

def _hash_int(*parts):
    raw = '|'.join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest(), 'big') % Q

def public_key(secret: int) -> int:
    return pow(G, secret % Q, P)

def prove(secret: int, identity: str, nonce: str = '', rng=None):
    """Create a proof. `rng` (a random.Random) is optional: Live mode passes its seeded
    RNG so a whole run is reproducible; Classic mode keeps using the `secrets` module."""
    secret %= Q
    r = rng.randint(1, Q - 1) if rng is not None else secrets.randbelow(Q - 1) + 1
    commitment = pow(G, r, P)
    challenge = _hash_int(identity, nonce, commitment) % Q
    response = (r + challenge * secret) % Q
    return {'commitment': commitment, 'challenge': challenge, 'response': response, 'nonce': nonce}

def verify(public: int, proof: dict, identity: str) -> bool:
    c = _hash_int(identity, proof.get('nonce',''), proof['commitment']) % Q
    lhs = pow(G, proof['response'], P)
    rhs = (proof['commitment'] * pow(public, c, P)) % P
    return c == proof['challenge'] and lhs == rhs
