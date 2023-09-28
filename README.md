# Unsupervised Domain Adaptation through the exploration of Temporal Consistency

## The problem
Deep Learning-based Semantic Segmentation has reached remarkable levels of accuracy throughout the years. Nonetheless, applying pretrained models to new domains---even though related---leads to considerable drops in performance due to domain shift. In addition to that, semantic labels are costly to obtain for new application scenarios, what makes the process of transfer learning even harder.
\par Lastly, since the majority of works in semantic segmentation rely on single-frame predictions, they miss a huge opportunity offered by reasoning on the temporal properties of video data. Ultimately, this leads to unstable perception models, which may harm overall performance and represent risks to the correct actuation of autonomous systems.

<div>
    <div style="float: left; width: 50%"><img src="/images/domain_shift.png" alt="Domain Shift" style="width:50%"><br><b>Domain shift</b></div>
    <div style="float: right; width: 50%"><img src="/images/stable_and_unstable_predictions.png" alt="Temporal Stability" style="width:40%"><br><b>Temporal (in)stability</b></div>
</div>

<!-- <div align="center"></div> <div align="center"></div> -->
<!-- </p>
<div class="row">
  <div class="column">
    <p align="center">
        <img src="/images/domain_shift.png" alt="Domain Shift" style="width:50%"><br>
        <b>Domain shift<b>
    </p>
  </div>
  <div class="column">
    <p align="center">
        <img src="/images/stable_and_unstable_predictions.png" alt="Temporal Stability" style="width:30%"><br>
        <b>Temporal (in)stability<b>
    </p>
  </div>
</div> -->

## Proposed solution
In light of that, this work addresses all previous problems by implementing a self-supervised auxiliary supervision strategy for learning temporal consistency in videos. The results show that, besides promoting temporal stability, our strategy greatly improves model precision in an Unsupervised Domain Adaptation (UDA) scenario.

<p align="center">
  <img src="/images/proposed_model.jpg">
  <b>Proposed architecture, built on top of a BiseNet V2 model.<b>
</p>

## Results

### Temporal Stability
<p align="center">
    <b>Before<b>
    <img src="/images/tc_before.jpg"><br>
    <b>After<b>
    <img src="/images/tc_after.jpg">
</p>

### Domain Adaptation
<p align="center">
    <img src="/images/domain_adaptation_results.png"><br>
    <b>Results of Domain Adaptation performed in a real-to-real scenario. We consider <a href="https://www.cityscapes-dataset.com/">Cityscapes</a> and <a href="https://bit.ly/zed2dataset">ZED2</a> as source and target datasets, respectively.<b>
</p>


## Related publications

For a more detailed description of the method and the technical training/testing aspects, please consider reading: 
* [Estudo de Estratégia de Aprendizado Auto-supervisionado para Aprimoramento da Consistência Temporal em Modelo de Segmentaçao Semântica Baseado em Deep Learning](https://doi.org/10.5753/semish.2023.230573)
* Soon, more ...

## Citation

If you find this project useful in your research, please consider citing:

```bibtex
@inproceedings{barbosa_and_osorio_semish_2023,
 author = {Felipe Barbosa and Fernando Osório},
 title = {Estudo de Estratégia de Aprendizado Auto-supervisionado para Aprimoramento da Consistência Temporal em Modelo de Segmentação Semântica Baseado em Deep Learning},
 booktitle = {Anais do L Seminário Integrado de Software e Hardware},
 location = {João Pessoa/PB},
 year = {2023},
 keywords = {},
 issn = {2595-6205},
 pages = {214--225},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {10.5753/semish.2023.230573},
 url = {https://sol.sbc.org.br/index.php/semish/article/view/25075}
}
```