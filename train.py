import argparse
import itertools

from allennlp.data.dataset_readers import Seq2SeqDatasetReader
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.nn.beam_search import BeamSearch
from build_modules import build_modules
from utils_data import *
from utils_nn import *


torch.manual_seed(1)

###### Hyper parameters ######
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=3, help='size of the batches')
parser.add_argument('--print_every', type=int, default=10, help='print frequency in batches')
parser.add_argument('--train_data', type=str, default='toy_data/train.txt', help='root directory of the train dataset')
parser.add_argument('--validation_data', type=str, default='toy_data/dev.tsv', help='root directory of the dev dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--cycle_loss_weight', type=int, default=1, help='weight of the cycle consistency loss')

parser.add_argument('--embedding_dim', type=int, default=300, help='dimention of all embeddings')
parser.add_argument('--hidden_dim', type=int, default=300, help='number of hidden units in all RNNs')
parser.add_argument('--max_decoding_steps', type=int, default=15, help='max decoding steps when we are missing targets')
parser.add_argument('--bidirectional', type=bool, default=True,
                    help='either to make all rnn bidirectional. Default is True')


# parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)
###################################

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Init ######
# init losses
criterion_GAN = torch.nn.MSELoss()
# cycle loss is currently computed inside the model

# create dataset reader
reader = Seq2SeqDatasetReader(source_tokenizer=WordTokenizer(word_splitter=JustSpacesWordSplitter()),
                              source_token_indexers={"ids": SingleIdTokenIndexer(namespace="language_A")},
                              target_token_indexers={"ids": SingleIdTokenIndexer(namespace="language_B")},
                              source_add_start_token=True)

# create datasets
train_dataset = reader.read(file_path=opt.train_data)
validation_dataset = reader.read(file_path=opt.validation_data)

# create vocabulary
vocab = Vocabulary.from_instances(train_dataset)

# create embeddings, generators, and discriminators
modules_dict = build_modules(vocab, opt)

# move to GPU
if opt.cuda():
    to_cuda(modules_dict)

# create optimizers
optimizer_generators = torch.optim.Adam(itertools.chain(modules_dict["generator_a2b"].parameters(),
                                                        modules_dict["generator_b2a"].parameters()),
                                        lr=opt.lr, betas=(0.5, 0.999))

optimizer_discriminators = torch.optim.Adam(itertools.chain(modules_dict["discriminator_a"].parameters(),
                                                            modules_dict["discriminator_b"].parameters()),
                                            lr=opt.lr, betas=(0.5, 0.999))

# create LR schedulers
lr_scheduler_generators = torch.optim.lr_scheduler.LambdaLR(optimizer_generators,
                                                            lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                               opt.decay_epoch).step)

lr_scheduler_discriminators = torch.optim.lr_scheduler.LambdaLR(optimizer_discriminators,
                                                                lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                   opt.decay_epoch).step)

# create batch generators
iterator = BasicIterator(batch_size=opt.batch_size)
iterator.index_with(vocab)
generator_train = iterator(train_dataset, shuffle=False, num_epochs=opt.n_epochs)
generator_validation = iterator(validation_dataset, shuffle=False)

# create golden tensors for discriminators
golden_target_real = torch.Tensor(opt.batch_size).fill_(1.0)
golden_target_fake = torch.Tensor(opt.batch_size).fill_(0.0)

# fixed data to check the progress on
iterator_dummy = BasicIterator(batch_size=1)
iterator_dummy.index_with(vocab)
generator_dummy = iterator_dummy(validation_dataset, shuffle=False)
fixed_batch = sample_new_batch(generator_dummy, vocab)
fixed_batch_A = prepare_model_input(source_tokens_dict=fixed_batch['source_tokens'])
fixed_batch_B = prepare_model_input(source_tokens_dict=fixed_batch['target_tokens'])

# beam search for inference
beam_search_A = BeamSearch(end_index=vocab.get_token_index(namespace="language_A", token='@end@'),
                           max_steps=opt.max_decoding_steps)
beam_search_B = BeamSearch(end_index=vocab.get_token_index(namespace="language_B", token='@end@'),
                           max_steps=opt.max_decoding_steps)
###################################


###### Training ######
curr_iter_train = 0
num_batches = iterator.get_num_batches(train_dataset)


for epoch_num in range(opt.epoch, opt.n_epochs):
    # for logging
    avg_cycle_losses = []
    avg_gan_losses = []
    discriminators_losses = []

    for batch_num in range(num_batches):
        # sample two unaligned training batches in language A and language B
        curr_batch = sample_new_batch(generator_train, vocab)
        real_batch_A = prepare_model_input(source_tokens_dict=curr_batch['source_tokens'])
        real_batch_B = prepare_model_input(source_tokens_dict=curr_batch['target_tokens'])

        # move to GPU
        if opt.cuda():
            to_cuda(real_batch_A)
            to_cuda(real_batch_B)

        ###### Generators ######
        optimizer_generators.zero_grad()

        # GAN loss A -> B
        fake_batch_B = modules_dict["generator_a2b"].forward(**real_batch_A)
        fake_batch_B = prepare_model_input(fake_batch_B)

        # teach discriminator B to say that the fake data is real (by updating generators weights)
        probs_fake_batch_B = modules_dict["discriminator_b"].forward(**fake_batch_B)
        loss_gan_a2b = criterion_GAN(probs_fake_batch_B, golden_target_real)

        # Cycle loss A -> B -> A
        # TODO: make pass to forward cleaner
        pseudo_parallel_batch_B2A = prepare_model_input(source_tokens_dict=fake_batch_B['source_tokens'],
                                                        target_tokens_dict=real_batch_A['source_tokens'])
        reconstructed_batch_A = modules_dict["generator_b2a"].forward(**pseudo_parallel_batch_B2A)
        loss_cycle_aba = reconstructed_batch_A['loss']

        # GAN loss B -> A
        # pass through generator
        fake_batch_A = modules_dict["generator_b2a"].forward(**real_batch_B)
        fake_batch_A = prepare_model_input(fake_batch_A)

        # teach discriminator A to say that the fake data is real (by updating generators weights)
        probs_fake_batch_A = modules_dict["discriminator_a"].forward(**fake_batch_A)
        loss_gan_b2a = criterion_GAN(probs_fake_batch_A, golden_target_real)

        # Cycle loss B -> A -> B
        # path through reverse generator
        pseudo_parallel_batch_A2B = prepare_model_input(source_tokens_dict=fake_batch_A['source_tokens'],
                                                        target_tokens_dict=real_batch_B['source_tokens'])
        reconstructed_batch_B = modules_dict["generator_a2b"].forward(**pseudo_parallel_batch_B2A)
        loss_cycle_bab = reconstructed_batch_B['loss']

        # Total generators loss
        loss_generators = loss_gan_a2b + opt.cycle_loss_weight * loss_cycle_aba + \
                          loss_gan_b2a + opt.cycle_loss_weight * loss_cycle_bab

        # Compute gradients for generators and discriminators
        loss_generators.backward()

        # Update generators and ignore discriminators
        optimizer_generators.step()
        ###################################

        ###### Discriminators ######
        optimizer_discriminators.zero_grad()

        # Discriminator A
        # teach discriminator_A to predict ones for real data from language A
        probs_real_batch_A = modules_dict["discriminator_a"].forward(**real_batch_A)
        loss_real_discriminator_a = criterion_GAN(probs_real_batch_A, golden_target_real)

        # teach discriminator_A to predict zeros for fake data from language A
        # TODO: implement experience reply
        fake_batch_A = detach_fake_batch(fake_batch_A)  # we do not update generators's parameters!
        probs_fake_batch_A = modules_dict["discriminator_a"].forward(**fake_batch_A)
        loss_fake_discriminator_a = criterion_GAN(probs_fake_batch_A, golden_target_fake)

        # total loss discriminator_A
        loss_total_discriminator_a = 0.5 * (loss_real_discriminator_a + loss_fake_discriminator_a)

        # Discriminator B
        # teach discriminator_B to predict ones for real data from language B
        probs_real_batch_B = modules_dict["discriminator_b"].forward(**real_batch_B)
        loss_real_discriminator_b = criterion_GAN(probs_real_batch_B, golden_target_real)

        # teach discriminator_B to predict zeros for fake data from language B
        # TODO: implement experience reply
        fake_batch_B = detach_fake_batch(fake_batch_B)  # we do not update generators's parameters!
        probs_fake_batch_B = modules_dict["discriminator_b"].forward(**fake_batch_B)  #
        loss_fake_discriminator_b = criterion_GAN(probs_fake_batch_B, golden_target_fake)

        # total loss discriminator_B
        loss_total_discriminator_b = 0.5 * (loss_real_discriminator_b + loss_fake_discriminator_b)

        # Compute total discriminators loss
        loss_discriminators = loss_total_discriminator_a + loss_total_discriminator_b

        # Compute gradients for discriminators only (we used detach to remove autograd history from generators)
        loss_discriminators.backward()

        # Update discriminators
        optimizer_discriminators.step()
        ###################################

        # Progress report
        if batch_num % opt.print_every == 0:
            avg_cycle_losses.append(((loss_cycle_bab + loss_cycle_aba) / 2.).item())
            avg_gan_losses.append(((loss_gan_a2b + loss_gan_b2a) / 2.).item())
            discriminators_losses.append(loss_discriminators.item())

            modules_dict['generator_a2b']._beam_search = beam_search_B
            modules_dict['generator_b2a']._beam_search = beam_search_A

            print("a -> b:")
            out = modules_dict['generator_a2b'].forward(**fixed_batch_A)
            modules_dict['generator_a2b'].decode(out)
            print(out["predicted_tokens"])

            print("b -> a:")
            out = modules_dict['generator_b2a'].forward(**fixed_batch_B)
            modules_dict['generator_b2a'].decode(out)
            print(out["predicted_tokens"])

            modules_dict['generator_a2b']._beam_search = None
            modules_dict['generator_b2a']._beam_search = None

            print("epoch #:", epoch_num,
                  "|| batch {}/{}:".format(batch_num, num_batches),
                  "|| discriminators_loss:", mean_of_list(discriminators_losses),
                  "|| avg_cycle_loss:", mean_of_list(avg_cycle_losses),
                  "|| avg_gan_loss:", mean_of_list(avg_gan_losses)
                  )

    # Update learning rates
    lr_scheduler_generators.step()
    lr_scheduler_discriminators.step()

    # Save models checkpoints
    torch.save(modules_dict['generator_a2b'].state_dict(), 'output/netG_A2B.pth')
    torch.save(modules_dict['generator_b2a'].state_dict(), 'output/netG_B2A.pth')
    torch.save(modules_dict['discriminator_a'].state_dict(), 'output/netD_A.pth')
    torch.save(modules_dict['discriminator_b'].state_dict(), 'output/netD_B.pth')