import pygame

pygame.init()

display_width = 1280
display_height = 720

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('viewer')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()
crashed = False

def display(image_path):
    gameDisplay.fill((0,0,0))
    carImg = pygame.image.load(image_path)
    gameDisplay.blit(carImg, (0, 0))

def load_images():
    file = open('all', 'r').read().split('\n')[:-1]
    return file

remove = []
images = load_images()
tick = 0
index = 0

while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if images[index] not in remove:
                    remove.append(images[index])

        if event.type == pygame.QUIT:
            crashed = True

    tick+=1
    if tick == 50:
        tick = 0
        if index == images.__len__():
            crashed = True
        else:
            index += 1
            try:
                display(images[index])
            except:
                print('error')

    pygame.display.update()
    clock.tick(60)

file = open('out', 'w')
for r in remove:
    file.write(r + '\n')

file.write(str(index))

pygame.quit()
quit()
