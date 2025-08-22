from PIL import Image
from tqdm import tqdm

for e in tqdm(range(10)):
    images = [Image.open(f"../WS_U_10k_test/output/event_{e}_angle_{n}.png") for n in range(360)]
    images[0].save(f'../WS_U_10k_test/event_{e}.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
