class Multimer(nn.module):
    def __init__(
        num_comb_blocks,
        num_heads,
        embed_dim,
        w_hidden,
        l_hidden,
        num_multimer_blocks,
    ):
        super(Multimer, self).__init__()
        self.pair2lat = nn.modulelist(Pair2Lat(embed_dim, num_heads) for _ in range(num_comb_blocks))
        self.lat2lat = nn.modulelist(All2Lat(embed_dim, num_heads) for _ in range(num_comb_blocks))

        
    def forward(self, monomer_latents: torch.tensor, pair: torch.tensor) -> torch.tensor:
        #in: monomer latents: (B, C, w, l, emb), pair: (B, sum(L), sum(L), emb)
        #out: (B, w, l, emb) - multimer latent with attention from multimer pair features and monomoer latents
        #creates multimer attention 

        self.get_latent(monomer_latents.shape[0], monomer_latents.shape[2], monomer_latents.shape[3], embed_dim=monomer_latents.shape[4])
        for pair2lat, lat2lat in zip(self.pair2lat, self.lat2lat):
            #perform attention flow from full pair to latent
            latent = self.pair2lat(latent, pair)
            #perform attention flow from monomer latents to latent, all happening simultaneously
            updated_latent = latent.clone()
            for l in range(len(monomer_latents.shape[1])): 
                updated_latent += self.lat2lat(latent, monomer_latents[:, l, :, :, :])
            l_shape = updated_latent.shape
            #normalize - the softmax applied at the end causes 
            latent = nn.functional.softmax(updated_latent.view(l_shape[0], -1, l_shape[3]), dim=1).view(l_shape)
        return latent



class Encoder(nn.module):
    def __init__(
        num_comb_blocks,
        num_heads,
        embed_dim,
        w_hidden,
        l_hidden,
        multimer_blocks,
    ):
    self.monomer = Monomer(num_comb_blocks, num_heads, embed_dim, w_hidden, l_hidden)
    self.multimer_blocks = nn.modulelist([Multimer() for _ in range(multimer_blocks)])
    def forward(self, seq: torch.tensor, pair: torch.tensor, lens:List[int]) -> torch.tensor:
        #in: seq: (B, C, L, emb), pair: (B, C, L, L, emb), lens: (B, C)
        #out: (B, w, l) - updated with attention from seq and pair
        #creates multimer attention 
        latents = torch.tensor([0, 0, 0, 0]) #placeholder, (B, C, w, l)
        idx = 0
        seq_pad
        for i in range(len(lens)):
            latents.cat(self.monomer(seq[:, i, :], pair[:, idx:idx+lens[i], idx:idx+lens[i], :]).unsqueeze(1), dim=1)
            idx += lens[i]

        for block in self.multimer_blocks:
            latent = block(latents, pair)
        return latent   